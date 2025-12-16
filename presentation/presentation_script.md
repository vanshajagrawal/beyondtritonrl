# Presentation Speaking Script
## Beyond TritonRL: RL for Triton Kernel Optimization

**Total Duration: 8-10 minutes**

---

## Slide 1: Title Slide
**Duration: 15 seconds**

"Good morning everyone. Today I'm presenting 'Beyond TritonRL: Reinforcement Learning for Triton Kernel Optimization with Modular Extensions.' My name is Vanshaj Agrawal, and I'll be discussing how I attempted to improve upon existing RL-based approaches for generating GPU kernels."

[Pause, then click to next slide]

---

## Slide 2: Introduction
**Duration: 45 seconds**

"To set the context: The performance of machine learning systems fundamentally depends on efficient GPU kernels. 

I begin by considering the KernelBench Benchmark. The core task is this: given a natural language description and a PyTorch reference implementation, generate a functionally correct AND performant Triton kernel. The challenge is particularly acute for Level 2 fusion tasks, where you're essentially asking the model to do the work of an expert compiler engineer."

I'm building upon TritonRL which represents the current state-of-the-art in using reinforcement learning to automatically generate Triton kernels from natural language descriptions.

However, the baseline has significant limitations. It achieves 63% correctness on basic operations like matrix multiplication, which is decent. But on more complex kernel fusion tasks—where I'm combining multiple operations like GEMM plus bias plus ReLU into a single efficient kernel—performance drops dramatically to just 7% correctness.

My goal was to improve this through four carefully designed independent extensions."

[Click to next slide]

---

## Slide 3: Problem Statement
**Duration: 45 seconds**

"This leads us to four key research questions:

First, can multi-input testing help the model generalize better instead of overfitting to single test cases?

Second, does staged evaluation—filtering out invalid candidates early—improve training efficiency?

Third, can adaptive curriculum learning enable better performance by building up from simple to complex tasks?

And fourth, how much does timing measurement noise actually affect RL training, and can I reduce it?

The core task is this: given a natural language description and a PyTorch reference implementation, generate a functionally correct AND performant Triton kernel. The challenge is particularly acute for Level 2 fusion tasks, where you're essentially asking the model to do the work of an expert compiler engineer."

[Click to next slide]

---

## Slide 4: Related Work
**Duration: 30 seconds**

"Before diving into my approach, let me briefly contextualize this work. Traditional approaches like TVM and Ansor use search-based optimization. Recent advances in RL for code generation, like AlphaCode, show promise for program synthesis.

The TritonRL baseline combines supervised fine-tuning with RL using best-of-N sampling—generating 10 candidates and keeping the 3 best. However, it has fragile verification using only single test cases, evaluates all candidates regardless of validity, uses a fixed task distribution, and relies on noisy single timing measurements.

My improvements address each of these weaknesses through independent, independently toggleable extensions."

[Click to next slide]

---

## Slide 5: System Architecture
**Duration: 45 seconds**

"Let me walk you through my system architecture. I use a two-phase training approach.

Phase 1 is supervised fine-tuning on the Qwen2.5-Coder-7B model using LoRA for parameter efficiency. I trained on 200 samples from the KernelBook dataset—a small subset to fit my budget and time constraints.

Phase 2 is the reinforcement learning phase. Here I generate 10 candidate kernels per task and keep the top 3 based on a reward function that considers both correctness and performance. Due to time constraints, I completed only 2 Level 2 fusion tasks rather than the planned 10.

Between the model and the reward computation sits my extension stack. This is the key innovation: four independent components that can be independently enabled or disabled for ablation studies. Multi-input testing generates diverse test cases. Staged evaluation filters candidates through five progressively expensive checks. Adaptive curriculum dynamically adjusts task difficulty. And calibrated timing reduces measurement noise through statistical protocols."

[Click to next slide]

---

## Slide 6: System Components
**Duration: 30 seconds**

"The system has a clean four-layer architecture. The model layer handles the actual neural network—Qwen with 8-bit quantization and LoRA adapters to fit on my GPU budget. The training orchestrator manages both SFT and RL phases across 8 A100 GPUs. The extension manager loads modules based on configuration. And the verification pipeline handles everything from code extraction to final performance measurement.

This independent design was deliberate—it allows me to clearly attribute improvements or issues to specific components, assuming I had the resources to run proper ablation studies."

[Click to next slide]

---

## Slide 7: Training Loop
**Duration: 30 seconds**

"Here's what happens in each training iteration. I sample a task from either Level 1 basics or Level 2 fusion. The model generates 10 candidates with temperature-controlled sampling for diversity. Each candidate flows through my extension stack—multi-input testing generates 5 test cases, staged evaluation applies the 5-stage filter, curriculum determines sampling ratios, and timing measures performance. I compute rewards combining correctness—worth up to 0.7—and a performance bonus up to 0.3. Finally, I select the top 3 candidates and update the model."

[Click to next slide]

---

## Slide 8: Extension 1 - Multi-Input Testing
**Duration: 30 seconds**

"Extension one tackles overfitting. The baseline uses a single test case, which allows models to memorize specific patterns rather than learning general correctness.

My solution generates five diverse test inputs for each kernel: the original input, scaled values at 2x, a different random seed, larger tensor dimensions, and different precision levels. This forces the model to handle edge cases and varying input patterns.

The benefit is straightforward—I catch errors that would slip through single-input testing. However, this also increases per-sample evaluation cost by 5x, creating a fundamental tradeoff between sample quality and sample quantity in resource-constrained settings."

[Click to next slide]

---

## Slide 9: Extension 2 - Staged Evaluation
**Duration: 30 seconds**

"Extension two addresses computational waste. Why spend minutes evaluating a kernel that won't even compile?

I designed a five-stage funnel. Stage 1 checks AST syntax in milliseconds, immediately rejecting malformed code. Stage 2 attempts compilation. Stage 3 runs on tiny 4-by-4 tensors. Stage 4 does full-scale testing. And only candidates that pass all these proceed to Stage 5's expensive timing measurements.

Candidates get partial reward credit based on how far they progress—0.3 for compiling, 0.5 for passing tiny runs, and so on. In unit testing, this gave me 35 to 40 percent throughput improvement by filtering invalid kernels early. However, partial credit may also flatten the reward landscape, potentially reducing learning signal."

[Click to next slide]

---

## Slide 10: Extension 3 - Adaptive Curriculum
**Duration: 30 seconds**

"Extension three implements progressive difficulty. The baseline uses a fixed mix of easy and hard tasks from the start, which can overwhelm the model before it has solid foundations.

My adaptive curriculum starts with only 10% Level 2 fusion tasks. As the model's Level 1 accuracy improves, I linearly increase the proportion of challenging tasks up to 50% once accuracy crosses the 40% threshold. This is shown in the formula here—basically, I build mastery at fundamentals before introducing complexity.

The logic is simple: you don't teach calculus before algebra. Though with only 200 training samples and 2 RL tasks, I didn't train long enough to observe curriculum effects."

[Click to next slide]

---

## Slide 11: Extension 4 - Calibrated Timing
**Duration: 30 seconds**

"Extension four tackles measurement noise. Single GPU timing measurements have about 15% variance due to thermal effects, scheduling, and other system noise. This makes it hard for the RL algorithm to learn what's actually faster.

My solution is a rigorous statistical protocol: 10 warmup runs to stabilize GPU state, 50 measurement trials using CUDA events for precision, and trimmed mean aggregation where I remove the top and bottom 10% of outliers.

The result? I reduced the coefficient of variation from 15% down to 5%—much more reliable signal for the RL algorithm. However, this also increases timing cost by 50x per sample."

[Click to next slide]

---

## Slide 12: Implementation
**Duration: 30 seconds**

"Let me quickly cover the implementation details. I used standard tools—Qwen2.5-Coder as the base model, LoRA for efficient fine-tuning, 8-bit quantization to fit my memory constraints, PyTorch as the framework, and Triton 2.1 as my target language.

Hardware was 8 A100 GPUs on AWS for both SFT and RL training. Total training time was approximately 20 minutes—5 minutes for SFT and 15 minutes for RL with 2 fusion tasks."

[Click to next slide]

---

## Slide 12b: Evaluation - Datasets and Protocol
**Duration: 45 seconds**

"Now let me be very explicit about my evaluation setup, since this is critical for understanding my results.

For training data, I used 200 samples from the KernelBook dataset, which contains 18,000 total Triton kernels. That's 1.1% of the available data. These 200 samples included 150 Level 1 basic kernels like matrix multiplication and softmax, and 50 Level 2 fusion kernels. My RL phase specifically targeted 2 fusion tasks: task zero was fused GEMM with bias and ReLU, task one was fused Conv2d with BatchNorm and ReLU.

For evaluation, I used a completely held-out test set: tasks 100 through 119 from the KernelBench suite—20 Level 2 fusion kernels that the model never saw during training. This is crucial for avoiding data leakage. Since I trained on tasks 0 and 1, testing on tasks 100-119 ensures I'm measuring true generalization, not memorization.

My primary metric is correct rate—the percentage of generated kernels that produce correct outputs when compared to PyTorch reference implementations. I also track valid rate, compiled rate, and speedup, but correctness is what matters most. A fast kernel that produces wrong answers is useless."

[Click to next slide]

---

## Slide 13: Results
**Duration: 45 seconds**

"Now for the results. Training completed successfully—my SFT loss decreased from 0.757 to 0.199 in 5 minutes. RL training completed 2 out of 2 fusion tasks in 15 minutes. RL fine-tuning achieved loss of 0.17 over 2 epochs. The pipeline ran without crashes.

However, the training scale was severely limited. I used only 200 samples—that's 1.1% of the full KernelBook dataset. I completed only 2 RL tasks instead of the planned 10. Total training time was 20 minutes compared to likely hours for the baseline.

Given this limited scale, I can't make definitive performance claims. Evaluation is currently running on the held-out test set. But this isn't trained at a scale comparable to the baseline, so direct comparison doesn't really make sense. This is a limitation of the study, not a fair evaluation of whether the extensions actually work."

[Pause for effect, then click to next slide]

---

## Slide 14: Why Limited Scale?
**Duration: 45 seconds**

"Why so limited? A few things.

Time constraints. Two hundred samples and 2 RL tasks took 20 minutes. Scaling to 1,000 samples and 10 tasks would've taken hours.

The sample efficiency tradeoff. My extensions increase cost per sample—5x for multi-input, 50x for timing. So it's 200 high-quality samples versus 1,000+ noisier ones. I chose quality, but maybe quantity matters more for RL.

Extension interactions. They looked good individually, but combined? I don't know without ablations, which I didn't have time for.

And pipeline overhead. Building the verification infrastructure took time, lots of debugging.

Bottom line: I don't know if extensions help or hurt at scale. That's the real limitation."

[Click to next slide]

---

## Slide 15: Lessons Learned
**Duration: 40 seconds**

"Despite the limited scale, I learned valuable scientific lessons.

First, verification infrastructure matters just as much as the model itself. The reliability of evaluation directly limits what you can learn from RL training. Code generation tasks need parser robustness comparable to production compilers. Verification bugs introduce measurement noise that can dominate learning signals.

Second, composability does not automatically follow from independentity. Extensions that individually improve metrics may interact negatively when combined. Reward shaping from multiple sources—like staged evaluation's partial credit plus performance bonuses—can create conflicting gradients. This is why ablation studies are necessary to establish causal attribution, not optional.

Third, there's a fundamental tradeoff between sample efficiency and verification rigor. Extensions that add robustness or precision increase per-sample cost 5 to 10 fold. In low-budget regimes, this creates a choice between fewer diverse samples with high-quality signal versus more samples with noisy signal. The optimal operating point depends on model capacity and task complexity, and I don't yet know which regime RL for code generation occupies.

Fourth, reward signal dilution in staged systems. Partial credit schemes may flatten the reward landscape by making 'nearly correct' kernels receive similar rewards to fully correct ones. An alternative hypothesis worth testing: use binary correctness rewards with curriculum over task difficulty, which may provide sharper gradients for learning."

[Click to next slide]

---

## Slide 16: Key Takeaways
**Duration: 40 seconds**

"Let me distill four key takeaways.

First, theoretical motivation does not equal empirical validation. My extensions were well-motivated—multi-input testing prevents overfitting, staged evaluation saves compute, curriculum learning builds progressively, calibrated timing reduces noise. All made sense. None were definitively validated at my training scale. This ambiguity itself is a scientific finding.

Second, limited compute creates attribution ambiguity. I cannot distinguish whether extensions hurt performance, or I simply didn't train long enough to see benefits, or I need different hyperparameters. Resolving this requires full-scale training with systematic ablations.

Third, ablation studies are not optional. Independent design enables ablations but doesn't replace them. I need to isolate individual extension contributions. With four binary extensions, that's 2 to the 4th power, or 16 configurations to test systematically.

Fourth, negative results still tell you something. Even inconclusive results guide future research. I've identified the sample efficiency versus verification rigor tradeoff, demonstrated the need for ablations before scaling, and shown that independent design enables future systematic study."

[Click to next slide]

---

## Slide 17: Contributions
**Duration: 30 seconds**

"So what did I actually contribute?

On the achievement side: I delivered complete end-to-end implementation of four independent extensions. I executed the full training pipeline successfully. I have results from a limited-scale run. I analyzed confounding factors like sample size and training duration. And I found some methodological lessons for RL-based code generation research.

What I cannot claim: The extensions did not improve over baseline, but I didn't train at comparable scale. I lack ablation studies to isolate individual contributions. And I cannot determine the root cause—was it the extensions themselves, insufficient training data, model choice, or training duration? All of these remain confounded.

So basically, I built and tested it, but the scale wasn't enough to draw strong conclusions."

[Click to next slide]

---

## Slide 18: Conclusion
**Duration: 30 seconds**

"To conclude: TritonRL achieves only 7% correctness on kernel fusion tasks. I proposed four independent extensions targeting specific weaknesses. Training succeeded but at severely limited scale—200 samples, 2 RL tasks, 20 minutes total. This is 1.1% of the baseline's data and a fraction of their training time.

Given this scale difference, I cannot make definitive claims about extension effectiveness. What I can say: theoretically sound extensions need validation at appropriate scale. Small-scale training cannot definitively evaluate multi-component systems. And ablation studies are necessary to establish causal attribution.

Future work requires full-scale training with 18K samples, 10+ RL tasks, and systematic ablation studies testing all 16 extension configurations."

[Click to next slide]

---

## Slide 19: Thank You
**Duration: 15 seconds**

"Thank you for your attention. The code is available at github.com/vanshajagrawal/beyondtritonrl if you'd like to examine or build upon my work. I'm happy to take questions."


---

## Q&A Tips

**If asked about future work:**
"The immediate next step would be full-scale training on the complete 18K dataset with systematic ablation studies. I'd need to test each extension independently—that's 16 configurations with 4 binary extensions. This would definitively determine if any extensions provide benefit at scale, and whether the issue was insufficient training or fundamental problems with the extensions themselves."

**If asked about compute budget:**
"I spent approximately $3 on AWS for this limited training run—20 minutes on 8 A100 GPUs. The severe scale reduction was due to time constraints, not just budget. Full replication with 1,000+ samples and 10 RL tasks would cost roughly $50-100 and take several hours. The baseline likely trained for many hours and cost hundreds of dollars. This scale difference makes direct comparison inappropriate."

**If asked why you think it failed:**
"I can't say it 'failed' because I didn't train at comparable scale. I just don't know if extensions help or hurt. The baseline used likely 18K samples over hours of training. I used 200 samples over 20 minutes. That's two orders of magnitude difference. Additionally, my extensions increase per-sample cost 5-50x, creating a fundamental tradeoff between sample quality and quantity. Without ablation studies at full scale, the question remains open."

**If asked what you'd do differently:**
"First, secure adequate compute budget and time upfront—at least enough to match baseline training scale. Second, run small-scale ablation studies early, testing each extension independently before combining them. Third, carefully analyze the sample efficiency versus verification rigor tradeoff—maybe the baseline's approach of noisy single measurements on many samples is actually optimal for RL. Fourth, implement A/B testing infrastructure from the start to enable rapid iteration."

**If asked about the scientific value:**
"Even inconclusive results are useful. I've got: one, a complete independent implementation that others can build on; two, identification of the sample efficiency versus verification rigor tradeoff, which is an important design choice for RL code generation systems; three, demonstration that composability doesn't follow from independent design—extensions need systematic interaction analysis; and four, documentation of what happens when you can't match baseline training scale, which is common but rarely reported."

---

## Timing Guidelines

- Slides 1-4 (Intro/Problem/Related): **2.5 minutes**
- Slides 5-7 (System Design): **1.75 minutes**
- Slides 8-11 (Extensions): **2 minutes**
- Slide 12 (Implementation): **0.5 minutes**
- **Slide 12b (Evaluation Details): 0.75 minutes** ← NEW, emphasizes datasets
- Slides 13-14 (Results/Scale): **1.5 minutes**
- Slides 15-18 (Lessons/Takeaways/Conclusion): **2.25 minutes**
- Slide 19 (Thank You): **0.25 minutes**

**Total: ~11.5 minutes** (leaves buffer for natural pacing and emphasis)

**To hit 8-10 minutes**:
- Reduce extensions (slides 8-11) to 20 seconds each: saves 40 seconds
- Speed through lessons learned: saves 30 seconds
- **Target: 10 minutes** (within acceptable range)

**Critical**: Spend full time on evaluation (slide 12b) - worth 4 points in grading rubric.
