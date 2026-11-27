use std::cmp::Ordering;
use std::collections::VecDeque;

/// 返回低于当前代际下限的条目位置，保持原容器的删除顺序。
pub(crate) fn lru_eviction_indices(generations: Vec<u64>, cutoff: u64) -> Vec<usize> {
    generations
        .into_iter()
        .enumerate()
        .filter_map(|(index, generation)| (generation < cutoff).then_some(index))
        .collect()
}

/// 按待执行节点顺序返回没有依赖阻塞的位置。
pub(crate) fn graph_ready_indices(block_counts: Vec<i64>) -> Vec<usize> {
    block_counts
        .into_iter()
        .enumerate()
        .filter_map(|(index, count)| (count == 0).then_some(index))
        .collect()
}

/// 剥离无依赖节点，保留环及被环阻塞的下游节点，顺序与 Python 原图一致。
pub(crate) fn graph_cycle_indices(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Option<Vec<usize>> {
    let mut incoming = vec![0usize; node_count];
    let mut outgoing = vec![Vec::new(); node_count];
    for &(from, to) in edges {
        outgoing.get_mut(from)?.push(to);
        *incoming.get_mut(to)? += 1;
    }

    let mut ready: VecDeque<_> = incoming
        .iter()
        .enumerate()
        .filter_map(|(node, &count)| (count == 0).then_some(node))
        .collect();
    while let Some(node) = ready.pop_front() {
        for &blocked in &outgoing[node] {
            incoming[blocked] -= 1;
            if incoming[blocked] == 0 {
                ready.push_back(blocked);
            }
        }
    }

    Some(
        incoming
            .into_iter()
            .enumerate()
            .filter_map(|(node, count)| (count != 0).then_some(node))
            .collect(),
    )
}

/// 按 RAM 回收评分和最后访问时间排序，仅接收 Python 已验证的有限数值。
pub(crate) fn ram_eviction_order(scores: &[(f64, f64)]) -> Vec<usize> {
    let compare = |left: f64, right: f64| {
        if left == right {
            Ordering::Equal
        } else {
            left.total_cmp(&right)
        }
    };
    let mut order: Vec<_> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| {
        let (left_score, left_time) = scores[left];
        let (right_score, right_time) = scores[right];
        compare(left_score, right_score).then_with(|| compare(left_time, right_time))
    });
    order
}

/// 计算每个空间或时间轴上分块的起点与长度，供 Python 依序切分张量。
pub(crate) fn tile_windows(
    shape: &[u64],
    tile: &[u64],
    overlap: &[u64],
) -> Option<Vec<Vec<(u64, u64)>>> {
    if shape.len() != tile.len() || shape.len() != overlap.len() {
        return None;
    }
    shape
        .iter()
        .zip(tile)
        .zip(overlap)
        .map(|((&extent, &size), &overlap)| {
            if extent <= size {
                return Some(vec![(0, extent)]);
            }

            let step = size.checked_sub(overlap)?;
            if step == 0 {
                return None;
            }
            let limit = extent - overlap;
            let count = (limit - 1) / step + 1;
            let mut windows = Vec::new();
            windows
                .try_reserve_exact(usize::try_from(count).ok()?)
                .ok()?;
            windows.extend((0..count).map(|index| {
                let start = index * step;
                (start, size.min(extent - start))
            }));
            Some(windows)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_retains_blocked_descendants() {
        // 环本身及其下游都与旧执行器一样留在未完成集合里。
        let edges = [(0, 1), (1, 0), (1, 3), (2, 3)];
        assert_eq!(graph_cycle_indices(4, &edges), Some(vec![0, 1, 3]));
        assert_eq!(graph_cycle_indices(3, &edges), None);
        assert_eq!(graph_ready_indices(vec![1, 0, 3, 0]), [1, 3]);
    }

    #[test]
    fn cache_plans_preserve_legacy_order() {
        // 没有达到淘汰年龄的条目不进入淘汰计划。
        assert_eq!(lru_eviction_indices(vec![0, 2, 1, 3], 2), [0, 2]);
        assert_eq!(
            ram_eviction_order(&[(2.0, 1.0), (1.0, 4.0), (2.0, 0.0)]),
            [1, 2, 0]
        );
    }

    #[test]
    fn tiled_axes_preserve_overlap_and_short_final_tiles() {
        // 图像轴末块缩短，覆盖整个视频时间轴时仍只生成一个块。
        assert_eq!(
            tile_windows(&[4, 11], &[8, 6], &[3, 2]),
            Some(vec![vec![(0, 4)], vec![(0, 6), (4, 6), (8, 3)]])
        );
        assert_eq!(tile_windows(&[9], &[6], &[6]), None);
    }
}
