const VRAM_ALIGNMENT: u128 = 1024;

/// 在计算批量布局时由形状和元素大小直接算出字段字节数。
pub(crate) fn tensor_geometry_bytes(
    mut shape: impl Iterator<Item = Option<u64>>,
    element_size: u64,
) -> Option<u128> {
    let elements = shape.try_fold(1_u128, |count, dimension| {
        count.checked_mul(u128::from(dimension?))
    })?;
    elements.checked_mul(u128::from(element_size))
}

/// 规划每个字段的数据范围。填充只影响下一个字段的起点。
pub(crate) fn gathered_byte_layout(
    sizes: impl Iterator<Item = Option<u128>>,
) -> Option<Vec<(u128, u128)>> {
    let mut start = 0_u128;
    let mut layout = Vec::with_capacity(sizes.size_hint().0);
    for size in sizes {
        let size = size?;
        layout.push((start, start.checked_add(size)?));
        let aligned = size.div_ceil(VRAM_ALIGNMENT).checked_mul(VRAM_ALIGNMENT)?;
        start = start.checked_add(aligned)?;
    }
    Some(layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_bytes_handle_scalars_zeros_and_overflow() {
        // 空形状仍占一个元素，零维长对应空张量，溢出时交回 Python。
        assert_eq!(tensor_geometry_bytes(std::iter::empty(), 8), Some(8));
        assert_eq!(
            tensor_geometry_bytes([3, 2, 4].into_iter().map(Some), 2),
            Some(48)
        );
        assert_eq!(
            tensor_geometry_bytes([3, 0, 4].into_iter().map(Some), 2),
            Some(0)
        );
        assert_eq!(
            tensor_geometry_bytes([u64::MAX; 3].into_iter().map(Some), 1),
            None
        );
        assert_eq!(
            tensor_geometry_bytes([u64::MAX; 2].into_iter().map(Some), 2),
            None
        );
    }

    #[test]
    fn alignment_applies_to_each_field() {
        // 一起对齐会漏掉第一个边界后的填充。
        let sizes = [0, 1, 1023, 1024, 1025].into_iter().map(Some);
        assert_eq!(
            gathered_byte_layout(sizes),
            Some(vec![
                (0, 0),
                (0, 1),
                (1024, 2047),
                (2048, 3072),
                (3072, 4097)
            ])
        );
        let large_sizes = [u128::from(u64::MAX), 1].into_iter().map(Some);
        assert_eq!(
            gathered_byte_layout(large_sizes),
            Some(vec![
                (0, u128::from(u64::MAX)),
                (1_u128 << 64, (1_u128 << 64) + 1)
            ])
        );
    }

    #[test]
    fn layout_ends_at_unpadded_payload() {
        // 最后一段只要求实际数据字节，零长度字段也保留自己的位置。
        assert_eq!(
            gathered_byte_layout([3, 0, 1025].into_iter().map(Some)),
            Some(vec![(0, 3), (1024, 1024), (1024, 2049)]),
        );
    }
}
