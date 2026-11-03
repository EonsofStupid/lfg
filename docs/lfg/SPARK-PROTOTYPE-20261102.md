# Spark-to-Windows developer prototype

On 2026-11-02 the owner's GB10 completed an owned 100,000-step pretraining
experiment and several subsequent assessment fine-tunes. The 369-record update
was exported to ONNX and embedded, with the developer UI, in a private
117,278,208-byte Windows x86-64 executable under the ignored local `target`
directory. This artifact contains learned weights derived from private data:
keep it local, do not commit or redistribute it, and treat its output as
experimental. The model has about 21.3 million parameters; it is not the
intended approximately 650 MB model.

The exported model's logits matched PyTorch across five selected input lengths
from 1 to 512 bytes, with a maximum absolute difference of 0.0000239. The
embedded native path matched PyTorch at those lengths within 0.0000134. A copy
of the executable in an otherwise empty directory passed an embedded-model
probe and served `/health`, the UI, the boundary-question catalog, and the
existing assessment endpoint. These are same-machine functional checks, not
a clean-machine or cross-platform qualification.

The assessment endpoint still uses the deterministic assessment implementation.
The trained model is available only through the experimental embedded byte
generator, so a person using the UI is not yet seeing the model's assessment.
The most recent separately evaluated 369-record checkpoint returned required
fields in four of ten reused development requests; five outputs were invalid
JSON and one omitted fields. This does not evaluate recommendation accuracy.

A small CPU diagnostic used one actual owner request with the configured
training framing, one warmup, three measured repetitions, and an eight-byte
generation limit. Median latency was 4.22 seconds per eight-byte response;
the measured process used 12.34 seconds of aggregate CPU time over the three
repetitions and reached a 266,387,456-byte peak working set. This is not a
full assessment, representative workload, cycles measurement, or a latency
qualification. It does establish that this implementation is not yet suited
to the intended low-latency CPU burst workload. The private report records
artifact and configuration digests.

The next prototype work is to implement a bounded, configurable assessment
path for the owned model, show its results and failures clearly in the UI,
collect real assessment outcomes and CPU measurements, and benchmark on
representative user hardware. Compare candidates with the deterministic
baseline and preserve a genuinely independent project or conversation test
set. Do not present generated explanations as verified evidence, infer model
availability, or automatically promote an update based on format compliance.
