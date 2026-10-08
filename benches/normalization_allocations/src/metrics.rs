use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

pub struct Allocator;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static PHASE: AtomicUsize = AtomicUsize::new(0);
static CALLS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static BYTES: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static FORWARD: AtomicUsize = AtomicUsize::new(0);
static TRANSPOSE: AtomicUsize = AtomicUsize::new(0);
static CLONES: AtomicUsize = AtomicUsize::new(0);
static PREDICTOR_SIZE: AtomicUsize = AtomicUsize::new(0);
static DESIGN_SIZE: AtomicUsize = AtomicUsize::new(0);
static PREDICTOR_CALLS: AtomicUsize = AtomicUsize::new(0);
static DESIGN_CALLS: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Relaxed) + bytes;
    if ENABLED.load(Relaxed) {
        PEAK_BYTES.fetch_max(live.saturating_sub(BASELINE_BYTES.load(Relaxed)), Relaxed);
        CALLS[0].fetch_add(1, Relaxed);
        BYTES[0].fetch_add(bytes, Relaxed);
        let phase = PHASE.load(Relaxed);
        if phase != 0 {
            CALLS[phase].fetch_add(1, Relaxed);
            BYTES[phase].fetch_add(bytes, Relaxed);
        }
        if bytes == PREDICTOR_SIZE.load(Relaxed) {
            PREDICTOR_CALLS.fetch_add(1, Relaxed);
        }
        if bytes == DESIGN_SIZE.load(Relaxed) {
            DESIGN_CALLS.fetch_add(1, Relaxed);
        }
    }
}

// SAFETY: All requests delegate unchanged to System, including alignment.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Relaxed);
        // SAFETY: The caller provides this pointer's original allocation layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller provides System's pointer, original layout, and valid size.
        let next = unsafe { System.realloc(pointer, layout, size) };
        if !next.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Relaxed);
            record(size);
        }
        next
    }
}

pub fn matrix_sizes(rows: usize, columns: usize) {
    PREDICTOR_SIZE.store(rows * columns * size_of::<f64>(), Relaxed);
    DESIGN_SIZE.store(rows * (columns + 1) * size_of::<f64>(), Relaxed);
}

pub fn forward() {
    if ENABLED.load(Relaxed) {
        FORWARD.fetch_add(1, Relaxed);
    }
}

pub fn transpose() {
    if ENABLED.load(Relaxed) {
        TRANSPOSE.fetch_add(1, Relaxed);
    }
}

struct Phase(usize);

impl Drop for Phase {
    fn drop(&mut self) {
        PHASE.store(self.0, Relaxed);
    }
}

fn in_phase<T>(phase: usize, operation: impl FnOnce() -> T) -> T {
    let _phase = Phase(PHASE.swap(phase, Relaxed));
    operation()
}

pub fn clone_vector<T>(operation: impl FnOnce() -> T) -> T {
    if ENABLED.load(Relaxed) {
        CLONES.fetch_add(1, Relaxed);
    }
    in_phase(1, operation)
}

pub fn statistics<T>(operation: impl FnOnce() -> T) -> T {
    in_phase(2, operation)
}

#[derive(Debug)]
pub struct Counts {
    pub peak_bytes: usize,
    pub elapsed_ns: u128,
    pub forward: usize,
    pub transpose: usize,
    pub clones: usize,
    pub calls: usize,
    pub bytes: usize,
    pub clone_calls: usize,
    pub clone_bytes: usize,
    pub stats_calls: usize,
    pub stats_bytes: usize,
    pub predictor_calls: usize,
    pub design_calls: usize,
}

pub fn measure<T>(operation: impl FnOnce() -> T) -> (T, Counts) {
    assert!(
        !ENABLED.load(Relaxed),
        "allocation measurements must be serial"
    );
    for counter in CALLS.iter().chain(&BYTES).chain([
        &FORWARD,
        &TRANSPOSE,
        &CLONES,
        &PREDICTOR_CALLS,
        &DESIGN_CALLS,
    ]) {
        counter.store(0, Relaxed);
    }
    BASELINE_BYTES.store(LIVE_BYTES.load(Relaxed), Relaxed);
    PEAK_BYTES.store(0, Relaxed);
    let start = std::time::Instant::now();
    ENABLED.store(true, Relaxed);
    let value = operation();
    ENABLED.store(false, Relaxed);
    let counts = Counts {
        peak_bytes: PEAK_BYTES.load(Relaxed),
        elapsed_ns: start.elapsed().as_nanos(),
        forward: FORWARD.load(Relaxed),
        transpose: TRANSPOSE.load(Relaxed),
        clones: CLONES.load(Relaxed),
        calls: CALLS[0].load(Relaxed),
        bytes: BYTES[0].load(Relaxed),
        clone_calls: CALLS[1].load(Relaxed),
        clone_bytes: BYTES[1].load(Relaxed),
        stats_calls: CALLS[2].load(Relaxed),
        stats_bytes: BYTES[2].load(Relaxed),
        predictor_calls: PREDICTOR_CALLS.load(Relaxed),
        design_calls: DESIGN_CALLS.load(Relaxed),
    };
    (value, counts)
}

impl Counts {
    pub fn print(
        &self,
        consumer: &str,
        backend: &str,
        fixture: &crate::fixture::Fixture,
        normalization: &str,
        phase: &str,
        iterations: usize,
    ) {
        let tracked = consumer == "shrinkage";
        let rejected = if tracked {
            self.forward.checked_sub(2 * self.transpose).unwrap()
        } else {
            0
        };
        let tracked_cell = |value: usize| {
            if tracked {
                value.to_string()
            } else {
                String::new()
            }
        };
        println!(
            "{consumer},{backend},{},{},{},{normalization},{phase},{iterations},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            fixture.x.nrows(),
            fixture.x.ncols(),
            fixture.density,
            tracked_cell(self.forward),
            tracked_cell(self.transpose),
            tracked_cell(rejected),
            tracked_cell(self.clones),
            self.calls,
            self.bytes,
            tracked_cell(self.clone_calls),
            tracked_cell(self.clone_bytes),
            tracked_cell(self.stats_calls),
            tracked_cell(self.stats_bytes),
            self.predictor_calls,
            self.design_calls,
            self.peak_bytes,
            self.elapsed_ns,
        );
    }
}
