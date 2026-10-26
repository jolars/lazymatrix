//! Peak live allocation measurement for single-threaded benchmark setup.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

struct Allocator;
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static BASELINE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

fn record(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    if ENABLED.load(Relaxed) {
        PEAK.fetch_max(live.saturating_sub(BASELINE.load(Relaxed)), Relaxed);
    }
}

// SAFETY: All pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: System allocated this pointer using the supplied layout.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller supplies System's original pointer and valid sizes.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            if size >= layout.size() {
                record(size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        result
    }
}

pub fn measure<T>(run: impl FnOnce() -> T) -> usize {
    assert!(!ENABLED.load(Relaxed));
    BASELINE.store(LIVE.load(Relaxed), Relaxed);
    PEAK.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let result = black_box(run());
    ENABLED.store(false, Relaxed);
    let peak = PEAK.load(Relaxed);
    drop(result);
    peak
}
