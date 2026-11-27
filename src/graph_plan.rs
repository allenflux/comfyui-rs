use std::collections::HashMap;

pub(crate) type InhibitionPlan = (Vec<usize>, Vec<(usize, usize)>);

/// 规划禁用节点时要删除的入边和要解除的下游依赖。
pub(crate) fn graph_inhibition_plan(
    node_count: usize,
    inhibited: &[usize],
    edges: &[(usize, usize)],
) -> Option<InhibitionPlan> {
    // 保留 Python 禁用集合的迭代顺序，用于重放图状态的修改。
    let mut inhibited_order = HashMap::with_capacity(inhibited.len());
    for (rank, &node) in inhibited.iter().enumerate() {
        if node >= node_count || inhibited_order.insert(node, rank).is_some() {
            return None;
        }
    }

    let mut source_order = HashMap::new();
    let mut incoming = Vec::new();
    let mut releases = vec![Vec::new(); inhibited.len()];
    for (edge_index, &(source, target)) in edges.iter().enumerate() {
        if source >= node_count || target >= node_count {
            return None;
        }
        let source_rank = *source_order.entry(source).or_insert(edge_index);
        match (inhibited_order.get(&source), inhibited_order.get(&target)) {
            (None, Some(&target_rank)) => incoming.push((source_rank, target_rank, edge_index)),
            (Some(&rank), None) => releases[rank].push((source, target)),
            _ => {}
        }
    }

    // 入边按来源顺序、随后按禁用集合顺序删除，解除依赖按来源分组。
    incoming.sort_unstable();
    Some((
        incoming.into_iter().map(|(_, _, index)| index).collect(),
        releases.into_iter().flatten().collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inhibition_keeps_source_and_removal_order() {
        // 禁用集合的顺序和目标边在图中的顺序不同。
        let edges = [
            (0, 1),
            (0, 4),
            (0, 2),
            (4, 5),
            (4, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (4, 0),
        ];
        assert_eq!(
            graph_inhibition_plan(6, &[4, 1], &edges),
            Some((vec![1, 0, 7], vec![(4, 5), (4, 0), (1, 2), (1, 3)]))
        );
    }

    #[test]
    fn invalid_indices_leave_the_graph_to_python() {
        // 索引和禁用列表有效时才允许调用方修改图。
        assert_eq!(graph_inhibition_plan(1, &[1], &[]), None);
        assert_eq!(graph_inhibition_plan(1, &[0, 0], &[]), None);
        assert_eq!(graph_inhibition_plan(1, &[], &[(0, 1)]), None);
        assert_eq!(
            graph_inhibition_plan(1, &[0], &[(0, 0)]),
            Some((vec![], vec![]))
        );
    }
}
