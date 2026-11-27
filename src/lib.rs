#[cfg(any(feature = "python", test))]
mod graph_plan;
#[cfg(any(feature = "python", test))]
mod model_memory;
#[cfg(any(feature = "python", test))]
mod planning;
#[cfg(feature = "python")]
mod scheduler;
#[cfg(any(feature = "python", test))]
mod tensor_layout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetAssessment {
    pub fits: bool,
    pub available_bytes: u64,
    pub shortfall_bytes: u128,
}

/// 根据 CUDA 驱动报告的空闲字节数，评估新增需求和安全余量能否同时满足。
pub fn assess_budget(
    free_bytes: u64,
    expected_extra_bytes: u64,
    headroom_bytes: u64,
) -> BudgetAssessment {
    // 用更宽的整数计算两个字节数之和，保留超出 u64 的精确缺口。
    let required_bytes = u128::from(expected_extra_bytes) + u128::from(headroom_bytes);
    let shortfall_bytes = required_bytes.saturating_sub(u128::from(free_bytes));

    BudgetAssessment {
        fits: shortfall_bytes == 0,
        available_bytes: free_bytes.saturating_sub(headroom_bytes),
        shortfall_bytes,
    }
}

/// 按 ComfyUI 的四项排序键确定候选顺序，完全相同的键保留输入顺序。
pub fn eviction_order(candidates: &[[i128; 4]]) -> Vec<usize> {
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        candidates[left]
            .cmp(&candidates[right])
            .then_with(|| left.cmp(&right))
    });
    order
}

#[cfg(feature = "python")]
mod python {
    use super::{graph_plan, model_memory, planning, scheduler, tensor_layout};
    use pyo3::prelude::*;
    use pyo3::types::{PyFloat, PyInt, PyList, PyTuple};

    /// 向 Python 返回预算是否满足、可用字节数和缺口字节数。
    #[pyfunction(name = "assess_budget")]
    fn assess_budget_py(
        free_bytes: u64,
        expected_extra_bytes: u64,
        headroom_bytes: u64,
    ) -> (bool, u64, u128) {
        let assessment = super::assess_budget(free_bytes, expected_extra_bytes, headroom_bytes);
        (
            assessment.fits,
            assessment.available_bytes,
            assessment.shortfall_bytes,
        )
    }

    /// 返回候选位置的排序结果，遇到 Python 自定义类型或超范围整数时交由 Python。
    #[pyfunction(name = "eviction_order")]
    fn eviction_order_py(candidates: &Bound<'_, PyAny>) -> Option<Vec<usize>> {
        let candidates = candidates.downcast_exact::<PyList>().ok()?;
        let mut keys = Vec::with_capacity(candidates.len());
        for candidate in candidates.iter() {
            let tuple = candidate.downcast_exact::<PyTuple>().ok()?;
            if tuple.len() != 4 {
                return None;
            }

            let mut key = [0; 4];
            for (part, value) in key.iter_mut().zip(tuple.iter()) {
                // 整数子类可能定义自己的比较规则，交回 Python 处理。
                let integer = value.downcast_exact::<PyInt>().ok()?;
                *part = integer.extract::<i128>().ok()?;
            }
            keys.push(key);
        }

        Some(super::eviction_order(&keys))
    }

    /// 只接收普通非负 Python 整数，保留自定义类型在原算法中的行为。
    fn unsigned(value: Bound<'_, PyAny>) -> Option<u64> {
        value.downcast_exact::<PyInt>().ok()?.extract().ok()
    }

    /// 读取来自模型或张量的原始尺寸，无法精确转换时回退。
    fn sizes(values: &Bound<'_, PyAny>) -> Option<Vec<u64>> {
        if let Ok(list) = values.downcast_exact::<PyList>() {
            return list.iter().map(unsigned).collect();
        }
        values
            .downcast_exact::<PyTuple>()
            .ok()?
            .iter()
            .map(unsigned)
            .collect()
    }

    /// 保留普通 Python 浮点数的运算类型，其他类型沿用 Python。
    fn float_value(value: &Bound<'_, PyAny>) -> Option<f64> {
        Some(value.downcast_exact::<PyFloat>().ok()?.value())
    }

    /// 返回可保留在低显存设备上的权重预算，保留 Python 的结果类型。
    #[pyfunction(name = "lowvram_load_budget")]
    fn lowvram_load_budget_py<'py>(
        py: Python<'py>,
        free_bytes: Bound<'py, PyAny>,
        loaded_bytes: Bound<'py, PyAny>,
        minimum_required: &Bound<'py, PyAny>,
        weight_ratio: &Bound<'py, PyAny>,
        minimum_inference: &Bound<'py, PyAny>,
    ) -> Option<Bound<'py, PyAny>> {
        match model_memory::lowvram_load_budget(
            unsigned(free_bytes)?,
            unsigned(loaded_bytes)?,
            float_value(minimum_required)?,
            float_value(weight_ratio)?,
            float_value(minimum_inference)?,
        )? {
            model_memory::LoadBudget::Integer(bytes) => {
                Some(bytes.into_pyobject(py).ok()?.into_any())
            }
            model_memory::LoadBudget::Float(bytes) => Some(PyFloat::new(py, bytes).into_any()),
        }
    }

    /// 根据预取流数量，一次计算顺序加载中的卸载窗口。
    #[pyfunction(name = "module_offload_windows")]
    fn module_offload_windows_py(
        offload_bytes: &Bound<'_, PyAny>,
        module_bytes: &Bound<'_, PyAny>,
        streams: Bound<'_, PyAny>,
    ) -> Option<Vec<u128>> {
        let streams = usize::try_from(unsigned(streams)?).ok()?;
        model_memory::module_offload_windows(&sizes(offload_bytes)?, &sizes(module_bytes)?, streams)
    }

    /// 选择动态加载的权重次序，四项键不能唯一确定顺序时回退。
    #[pyfunction]
    fn model_loading_order(keys: Vec<(bool, i128, i128, String)>) -> Option<Vec<usize>> {
        model_memory::model_loading_order(&keys)
    }

    /// 从 LRU 代际快照批量选择要淘汰的缓存位置。
    #[pyfunction]
    fn lru_eviction_indices(generations: Vec<u64>, cutoff: u64) -> Vec<usize> {
        planning::lru_eviction_indices(generations, cutoff)
    }

    /// 根据内存压力评分和访问时间计算回收次序。
    #[pyfunction]
    fn ram_eviction_order(scores: Vec<(f64, f64)>) -> Vec<usize> {
        planning::ram_eviction_order(&scores)
    }

    /// 从工作流元数据中提取没有阻塞依赖的节点。
    #[pyfunction]
    fn graph_ready_indices(block_counts: Vec<i64>) -> Vec<usize> {
        planning::graph_ready_indices(block_counts)
    }

    /// 按原顺序返回发生循环依赖时无法执行的节点。
    #[pyfunction]
    fn graph_cycle_indices(node_count: usize, edges: Vec<(usize, usize)>) -> Option<Vec<usize>> {
        planning::graph_cycle_indices(node_count, &edges)
    }

    /// 从现有边批量规划控制节点要移除的连接和释放的依赖。
    #[pyfunction]
    fn graph_inhibition_plan(
        node_count: usize,
        inhibited: Vec<usize>,
        edges: Vec<(usize, usize)>,
    ) -> Option<graph_plan::InhibitionPlan> {
        graph_plan::graph_inhibition_plan(node_count, &inhibited, &edges)
    }

    /// 为图像和视频返回空间轴及时间轴的分块窗口。
    #[pyfunction]
    fn tile_windows(
        shape: &Bound<'_, PyAny>,
        tile: &Bound<'_, PyAny>,
        overlap: &Bound<'_, PyAny>,
    ) -> Option<Vec<Vec<(u64, u64)>>> {
        planning::tile_windows(&sizes(shape)?, &sizes(tile)?, &sizes(overlap)?)
    }

    /// 普通张量直接传字节数，几何模板传形状和元素大小，避免逐个调用。
    fn gathered_field_size(field: Bound<'_, PyAny>) -> Option<u128> {
        if let Ok(size) = field.downcast_exact::<PyInt>() {
            return size.extract::<u64>().ok().map(u128::from);
        }

        let geometry = field.downcast_exact::<PyTuple>().ok()?;
        if geometry.len() != 2 {
            return None;
        }
        let shape = geometry.get_item(0).ok()?;
        let dimensions = shape.downcast_exact::<PyTuple>().ok()?.iter();
        let element_size = unsigned(geometry.get_item(1).ok()?)?;
        tensor_layout::tensor_geometry_bytes(dimensions.map(unsigned), element_size)
    }

    /// 一次计算所有字段的原始字节数、对齐和数据范围。
    #[pyfunction]
    fn gathered_byte_layout(fields: &Bound<'_, PyAny>) -> Option<Vec<(u128, u128)>> {
        if let Ok(list) = fields.downcast_exact::<PyList>() {
            return tensor_layout::gathered_byte_layout(list.iter().map(gathered_field_size));
        }
        let tuple = fields.downcast_exact::<PyTuple>().ok()?;
        tensor_layout::gathered_byte_layout(tuple.iter().map(gathered_field_size))
    }

    /// 注册 Python 接入层使用的原生函数。
    #[pymodule]
    fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(assess_budget_py, module)?)?;
        module.add_function(wrap_pyfunction!(eviction_order_py, module)?)?;
        module.add_class::<scheduler::PromptScheduler>()?;
        module.add_class::<scheduler::PromptSubmissionGate>()?;
        module.add_function(wrap_pyfunction!(lowvram_load_budget_py, module)?)?;
        module.add_function(wrap_pyfunction!(module_offload_windows_py, module)?)?;
        module.add_function(wrap_pyfunction!(model_loading_order, module)?)?;
        module.add_function(wrap_pyfunction!(lru_eviction_indices, module)?)?;
        module.add_function(wrap_pyfunction!(ram_eviction_order, module)?)?;
        module.add_function(wrap_pyfunction!(graph_ready_indices, module)?)?;
        module.add_function(wrap_pyfunction!(graph_cycle_indices, module)?)?;
        module.add_function(wrap_pyfunction!(graph_inhibition_plan, module)?)?;
        module.add_function(wrap_pyfunction!(tile_windows, module)?)?;
        module.add_function(wrap_pyfunction!(gathered_byte_layout, module)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fit_preserves_headroom() {
        // 允许任务恰好用满扣除安全余量后的预算。
        let assessment = assess_budget(10, 8, 2);
        assert_eq!(
            assessment,
            BudgetAssessment {
                fits: true,
                available_bytes: 8,
                shortfall_bytes: 0,
            }
        );
    }

    #[test]
    fn reports_bytes_missing_after_headroom() {
        // 缺口包含给安全余量保留的字节。
        let assessment = assess_budget(10, 9, 2);
        assert!(!assessment.fits);
        assert_eq!(assessment.available_bytes, 8);
        assert_eq!(assessment.shortfall_bytes, 1);
    }

    #[test]
    fn insufficient_headroom_fails_even_without_incremental_demand() {
        // 已经低于安全余量时，零新增需求也不能满足预算。
        let assessment = assess_budget(1, 0, 3);
        assert!(!assessment.fits);
        assert_eq!(assessment.available_bytes, 0);
        assert_eq!(assessment.shortfall_bytes, 2);
    }

    #[test]
    fn adding_large_inputs_does_not_overflow() {
        // 覆盖两个 u64 输入量相加超出 u64 的边界。
        let assessment = assess_budget(0, u64::MAX, u64::MAX);
        assert!(!assessment.fits);
        assert_eq!(assessment.shortfall_bytes, 2 * u128::from(u64::MAX));
    }

    #[test]
    fn eviction_order_uses_all_four_keys_and_preserves_equal_keys() {
        // 让每个排序键都实际参与比较，并覆盖相等键与整数端点。
        let candidates = [
            [-10, 4, 8, 3],
            [-10, 4, 8, 3],
            [-20, 5, 10, 4],
            [-10, 3, 9, 2],
            [-10, 4, 7, 1],
            [-10, 4, 8, 0],
            [i128::MAX, 0, 0, 0],
            [i128::MIN, 0, 0, 0],
        ];

        assert_eq!(eviction_order(&candidates), [7, 2, 3, 4, 5, 0, 1, 6]);
        assert!(eviction_order(&[]).is_empty());
    }
}
