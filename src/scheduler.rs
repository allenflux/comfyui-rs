use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use pyo3::prelude::*;

struct Pending {
    priority: f64,
    prompt_id: String,
    ticket: u64,
}

impl Ord for Pending {
    /// 复制 ComfyUI 堆的任务编号和标识排序规则。
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| self.prompt_id.cmp(&other.prompt_id))
            .then_with(|| self.ticket.cmp(&other.ticket))
    }
}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Pending {}

/// 只持有调度键和句柄，Python 负责存放任务和它的状态。
#[pyclass(module = "comfyui_rs._native")]
#[derive(Default)]
pub(crate) struct PromptScheduler {
    pending: BinaryHeap<Reverse<Pending>>,
    identities: HashSet<(u64, String)>,
}

#[pymethods]
impl PromptScheduler {
    /// 创建独立的 worker 任务堆。
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// 入队并保留原始编号顺序。重复键需要由 Python 比较剩余任务字段。
    fn push(&mut self, ticket: u64, priority: f64, prompt_id: String) -> bool {
        if !priority.is_finite() {
            return false;
        }
        // Python 把正负零视为同一个优先级。
        let priority = if priority == 0.0 { 0.0 } else { priority };
        if !self
            .identities
            .insert((priority.to_bits(), prompt_id.clone()))
        {
            return false;
        }

        self.pending.push(Reverse(Pending {
            priority,
            prompt_id,
            ticket,
        }));
        true
    }

    /// 取出下一项，并释放该条目的排序键。
    fn pop(&mut self) -> Option<u64> {
        let Reverse(entry) = self.pending.pop()?;
        self.identities
            .remove(&(entry.priority.to_bits(), entry.prompt_id));
        Some(entry.ticket)
    }

    /// 删除尚未执行的任务句柄，同时更新对应的排序键。
    fn remove(&mut self, ticket: u64) -> bool {
        let mut removed = None;
        self.pending.retain(|Reverse(entry)| {
            if entry.ticket == ticket {
                removed = Some((entry.priority.to_bits(), entry.prompt_id.clone()));
                false
            } else {
                true
            }
        });
        removed.is_some_and(|identity| self.identities.remove(&identity))
    }

    /// 清空待执行任务，允许重新使用原排序键。
    fn clear(&mut self) {
        self.pending.clear();
        self.identities.clear();
    }

    /// 为 Python 队列快照返回堆中的任务句柄。
    fn tickets(&self) -> Vec<u64> {
        self.pending
            .iter()
            .map(|Reverse(entry)| entry.ticket)
            .collect()
    }

    /// 返回尚未开始执行的任务数。
    fn __len__(&self) -> usize {
        self.pending.len()
    }
}

/// 统计正在读取和验证的提交请求，许可只占用本进程的并发名额。
#[pyclass(module = "comfyui_rs._native")]
pub(crate) struct PromptSubmissionGate {
    limit: usize,
    acquired: AtomicUsize,
}

impl PromptSubmissionGate {
    /// 按当前计数尝试更新，遇到另一线程修改时用最新值重算。
    fn update_count(&self, next: impl Fn(usize) -> Option<usize>) -> bool {
        let mut count = self.acquired.load(AtomicOrdering::Relaxed);
        while let Some(updated) = next(count) {
            match self.acquired.compare_exchange_weak(
                count,
                updated,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(observed) => count = observed,
            }
        }
        false
    }
}

#[pymethods]
impl PromptSubmissionGate {
    /// 按配置建立提交上限，计数从零开始。
    #[new]
    fn new(limit: usize) -> Self {
        Self {
            limit,
            acquired: AtomicUsize::new(0),
        }
    }

    /// 一次原子操作检查并占用名额，已满时立即返回。
    fn try_acquire(&self) -> bool {
        self.update_count(|count| (count < self.limit).then(|| count + 1))
    }

    /// 请求结束或取消时归还名额，防止计数从零下溢。
    fn release(&self) {
        self.update_count(|count| count.checked_sub(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_by_number_then_prompt_id() {
        // 显式插队优先于普通任务，同一优先级按任务标识排序。
        let mut scheduler = PromptScheduler::new();
        assert!(scheduler.push(1, 2.0, "b".into()));
        assert!(scheduler.push(2, -1.0, "c".into()));
        assert!(scheduler.push(3, 2.0, "a".into()));
        assert_eq!(scheduler.__len__(), 3);
        assert_eq!(scheduler.pop(), Some(2));
        assert_eq!(scheduler.pop(), Some(3));
        assert_eq!(scheduler.pop(), Some(1));
        assert_eq!(scheduler.pop(), None);
    }

    #[test]
    fn duplicate_and_invalid_keys_leave_heap_unchanged() {
        // 相同的键不改变堆，取消与清空后可重新使用该键。
        let mut scheduler = PromptScheduler::new();
        assert!(scheduler.push(1, -0.0, "a".into()));
        assert!(!scheduler.push(2, 0.0, "a".into()));
        assert!(!scheduler.push(3, f64::NAN, "a".into()));
        assert_eq!(scheduler.tickets(), [1]);
        assert!(scheduler.remove(1));
        assert!(scheduler.push(4, 0.0, "a".into()));
        scheduler.clear();
        assert!(scheduler.push(5, -0.0, "a".into()));
        assert_eq!(scheduler.tickets(), [5]);
    }

    #[test]
    fn submission_slots_enforce_limits_and_allow_retry() {
        // 已满的请求不能占用名额，归还后可以立即重试。
        let gate = PromptSubmissionGate::new(2);
        gate.release();
        assert!(gate.try_acquire());
        assert!(gate.try_acquire());
        assert!(!gate.try_acquire());
        gate.release();
        assert!(gate.try_acquire());
        assert!(!gate.try_acquire());
    }
}
