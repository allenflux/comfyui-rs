/// 保留 Python 在预算为零前后使用的整数和浮点数类型。
pub(crate) enum LoadBudget {
    Integer(i64),
    Float(f64),
}

/// 计算低显存模型可继续驻留的权重字节，保留原规则的比较顺序。
pub(crate) fn lowvram_load_budget(
    free: u64,
    loaded: u64,
    minimum_required: f64,
    weight_ratio: f64,
    minimum_inference: f64,
) -> Option<LoadBudget> {
    // 未能精确转换为浮点或包含特殊浮点值时，由 Python 使用原算法。
    if free > (1 << 53)
        || loaded > (1 << 53)
        || !minimum_required.is_finite()
        || !weight_ratio.is_finite()
        || !minimum_inference.is_finite()
    {
        return None;
    }

    let free = free as f64;
    let reserved = free - minimum_required;
    let ratio = free * weight_ratio;
    let inference = free - minimum_inference;
    let preference = if inference < ratio { inference } else { ratio };
    if reserved <= 0.0 && preference <= 0.0 {
        return Some(if loaded == 0 {
            LoadBudget::Float(0.1)
        } else {
            LoadBudget::Integer(-(loaded as i64))
        });
    }

    let budget = if preference > reserved {
        preference
    } else {
        reserved
    } - loaded as f64;
    Some(LoadBudget::Float(if budget == 0.0 { 0.1 } else { budget }))
}

/// 用滑动窗口累计随后几个权重模块的卸载需求，保留整数精度。
pub(crate) fn module_offload_windows(
    offload: &[u64],
    modules: &[u64],
    streams: usize,
) -> Option<Vec<u128>> {
    if offload.len() != modules.len() {
        return None;
    }
    if streams == 0 {
        return Some(offload.iter().map(|&bytes| u128::from(bytes)).collect());
    }

    let mut sum: u128 = modules
        .iter()
        .skip(1)
        .take(streams)
        .map(|&size| u128::from(size))
        .sum();
    let mut result = Vec::with_capacity(offload.len());
    for (index, &bytes) in offload.iter().enumerate() {
        result.push(u128::from(bytes) + sum);
        if let Some(&leaving) = modules.get(index + 1) {
            sum -= u128::from(leaving);
            if let Some(&entering) = modules.get(index.saturating_add(streams).saturating_add(1)) {
                sum += u128::from(entering);
            }
        }
    }
    Some(result)
}

/// 按 ComfyUI 的动态模型权重键排序，相同前缀交给 Python 比较剩余对象字段。
pub(crate) fn model_loading_order(keys: &[(bool, i128, i128, String)]) -> Option<Vec<usize>> {
    let mut order: Vec<_> = (0..keys.len()).collect();
    order.sort_unstable_by(|&left, &right| keys[left].cmp(&keys[right]));
    if order.windows(2).any(|pair| keys[pair[0]] == keys[pair[1]]) {
        return None;
    }
    Some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_preserve_original_zero_and_negative_cases() {
        // Python 将零视为全量加载的特殊值，所以继续用 0.1 区分。
        assert!(matches!(
            lowvram_load_budget(100, 50, 50.0, 0.4, 30.0),
            Some(LoadBudget::Float(0.1))
        ));
        assert!(matches!(
            lowvram_load_budget(100, 50, 110.0, 0.0, 110.0),
            Some(LoadBudget::Integer(-50))
        ));
        assert!(matches!(
            lowvram_load_budget(100, 50, 80.0, 0.0, 110.0),
            Some(LoadBudget::Float(-30.0))
        ));
    }

    #[test]
    fn offload_windows_do_not_include_current_module_size() {
        // 当前模块已有独立卸载成本，窗口只累计后续模块。
        assert_eq!(
            module_offload_windows(&[10, 20, 30], &[1, 2, 3], 2),
            Some(vec![15, 23, 30])
        );
        assert_eq!(module_offload_windows(&[10], &[1, 2], 1), None);
        assert_eq!(
            module_offload_windows(&[u64::MAX, 0], &[0, 1], 1),
            Some(vec![u128::from(u64::MAX) + 1, 0])
        );
    }

    #[test]
    fn dynamic_loading_respects_weight_group_and_full_key() {
        // 动态加载按上游的大小分组和权重顺序排序，同前缀留给原对象比较。
        let first = (false, -20, 20, String::from("block.0"));
        let second = (false, -15, 15, String::from("block.1"));
        let large = (true, -90, 90, String::from("block.2"));
        assert_eq!(
            model_loading_order(&[large, second, first.clone()]),
            Some(vec![2, 1, 0])
        );
        assert_eq!(model_loading_order(&[first.clone(), first]), None);
    }
}
