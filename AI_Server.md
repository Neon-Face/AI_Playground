## Catalog
1. [[#^c2f2c4|Purpose of the AI server]]
2. [[#^64f480|Choosing LLM]]
3. [[#^3be55f|Hardware requirements]]
4. [[#^8e2aaa|Solutions]]
5. [[#^fd7fe0|Other Information]]
6. [[#^4a397c|References]]
## Purpose of the AI Server

^c2f2c4

Start by defining your goal, not by looking at hardware. The specific tasks you plan to run on your AI server will directly determine your hardware costs. A key factor to consider is whether you'll be fine-tuning models or primarily running inference, as their VRAM demands are vastly different. This is further explained in [[#^a4ad18|A VRAM Showdown Between Fine-Tuning and Inference for a 30B Local LLM]].

#### Fine-tuning
- Goal: To adapt an already pre-trained LLM (base model) to a specific task, domain, or style, using a targeted, relatively small dataset for secondary training, thereby improving performance in niche scenarios.
- Computational Characteristics: Highly compute-intensive. Involves updating some or all model weights, requiring back propagation and gradient calculation.
- Resource Requirements: High. The primary bottleneck is VRAM capacity, as model parameters, optimizer states, and activations need to be loaded into VRAM. Strong computational power is also needed to accelerate training. VRAM requirements can be reduced through multi-GPU setups or efficient fine-tuning methods (e.g., LoRA/QLoRA).
- Typical Executors: Developers, researchers, enterprise users looking to customize LLMs for specific needs.
#### Inference
- Goal: To use an already trained model with fixed weights to receive user input (Prompt) and generate output (e.g., text, code), for practical application.
- Computational Characteristics: Relatively less compute-intensive, involving only the forward pass of the model, with no weight updates. Key metric is generation speed (Tokens/second).
- Resource Requirements: Medium to High. Primarily depends on model size and quantization level. Requires loading model parameters into VRAM and relies on high computational power and VRAM bandwidth for fast output generation. Quantization techniques can significantly reduce VRAM requirements.
- Typical Executors: Companies offering LLM services, users deploying local LLMs.

---
## Choosing LLM

^64f480

In my use case, I mainly focus on agentic capability, using Gemini 2.5 Flash as my reasoning benchmark. While benchmarks from [Artificial Analysis](https://artificialanalysis.ai/models/gemini-2-5-flash-reasoning?models=gpt-oss-120b%2Cgpt-oss-20b%2Cgemini-2-5-flash-reasoning%2Cgemini-2-5-pro%2Cmagistral-small-2509%2Cdeepseek-v3-1%2Cdeepseek-v3-1-reasoning%2Cdeepseek-r1%2Cqwen3-30b-a3b-2507-reasoning%2Cqwen3-30b-a3b-2507%2Cqwen3-4b-2507-instruct-reasoning) and [LiveBench](https://livebench.ai/#/?organization=Alibaba%2CDeepSeek%2CGoogle%2CMistral+AI%2CMeta) (up to September 25, 2025) pointed to the strength of the Qwen3 series, my hands-on testing was the real deciding factor. After using `Qwen3-4B-Instruct-2507` (8-bit), I was incredibly impressed by its performance as it flawlessly executed my instructions and tool calls within the Google ADK framework. This experience really underscores why it's crucial to supplement benchmarks by watching real-world videos and community posts, since different models excel at different tasks. Seeing such remarkable capability from a 4B model gives me high confidence in its larger counterparts, so I am now proceeding to set up my hardware for [Qwen3-30B-A3B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8).
![[Qwen3-30B-A3B-Thinking-2507.jpeg]]

![[Pasted image 20250925135724.png]]

| Model                           | Global Average | Reasoning Average | Coding Average | Agentic Coding Average | Mathematics Average | Data Analysis Average | Language Average | IF Average |
| :------------------------------ | :------------- | :---------------- | :------------- | :--------------------- | :------------------ | :-------------------- | :--------------- | :--------- |
| Gemini 2.5 Pro (Max Thinking)   | 70.95          | 94.28             | 73.90          | 20.00                  | 84.19               | 71.50                 | 75.44            | 77.35      |
| Qwen 3 235B A22B Thinking 2507  | 70.76          | 91.56             | 67.18          | 20.00                  | 81.14               | 74.65                 | 70.86            | 89.96      |
| DeepSeek V3.1 Thinking          | 70.75          | 90.67             | 70.31          | 25.00                  | 88.72               | 64.33                 | 70.40            | 85.85      |
| DeepSeek R1 (2025-05-28)        | 70.10          | 91.08             | 71.40          | 26.67                  | 85.26               | 71.54                 | 64.82            | 79.95      |
| DeepSeek V3.1 Terminus Thinking | 69.97          | 85.61             | 71.40          | 20.00                  | 89.28               | 71.76                 | 69.46            | 82.28      |
| Gemini 2.5 Pro                  | 69.39          | 93.72             | 70.70          | 13.33                  | 83.33               | 71.60                 | 74.52            | 78.54      |
| Qwen 3 Next 80B A3B Instruct    | 66.54          | 88.22             | 68.20          | 11.67                  | 80.67               | 68.63                 | 68.17            | 80.20      |
| Qwen 3 Next 80B A3B Thinking    | 65.29          | 91.25             | 60.66          | 10.00                  | 82.37               | 73.16                 | 54.48            | 85.08      |
| DeepSeek R1                     | 65.15          | 77.17             | 76.07          | 20.00                  | 77.91               | 69.63                 | 54.77            | 80.51      |
| Qwen 3 235B A22B Thinking       | 64.93          | 77.94             | 66.41          | 13.33                  | 80.15               | 68.31                 | 60.61            | 87.73      |
| Qwen 3 235B A22B Instruct 2507  | 64.72          | 86.89             | 66.41          | 13.33                  | 79.18               | 65.24                 | 66.29            | 75.70      |
| Gemini 2.5 Flash                | 64.42          | 78.53             | 63.53          | 18.33                  | 84.10               | 69.85                 | 57.04            | 79.56      |
![[Pasted image 20250926101708.png]]

Deciding on a quantized model is a critical step that balances VRAM requirements, cost, and performance. With the [official Qwen quantization benchmarks](https://qwen.readthedocs.io/en/latest/getting_started/quantization_benchmark.html#performance-of-quantized-models) pending an update (as of September 25, 2025), we can turn to recent, in-depth studies on models like DeepSeek to make an informed choice.
According to a [Quantitative Analysis of DeepSeek](https://arxiv.org/html/2505.02390v1) and a large-scale [Red Hat evaluation](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms), the consensus is clear: modern quantization can significantly reduce resource needs with minimal, often negligible, performance loss.
For a balanced approach, 4-bit quantization (like Q4_K_M) stands out as the optimal choice. The DeepSeek analysis shows its 4-bit model maintains performance nearly identical to the full FP8 version, while the Red Hat study confirms 4-bit models recover almost 99% of baseline accuracy, all while making the model ~3.5x smaller. If your hardware allows and maximum fidelity is the goal, 8-bit quantization is an even safer bet, consistently recovering over 99% of the original model's accuracy. For scenarios with extreme memory constraints, advanced 3-bit methods can perform on par with 4-bit, but be wary of standard implementations which can cause a noticeable drop. 
Ultimately, the evidence suggests larger models are more resilient to quantization, but for most users, a well-implemented 4-bit model delivers the best combination of cost-efficiency and high-fidelity performance. After deciding, you can use the [LLM Inference VRAM Calculator](https://apxml.com/tools/vram-calculator) to estimate your hardware needs precisely.

![[Pasted image 20250929142517.png]]
## Hardware requirements

^3be55f

### Key factors:
#### VRAM: 

>the model must fit in VRAM or it won't load, and relying on system RAM will severely degrade performance.

Roughly 2 GB of VRAM per billion parameters at FP16.

![[Pasted image 20250925100518.png]]
#### Memory Bandwidth

>Memory bandwidth (GB/s) is how quickly a GPU can move data within VRAM. It directly affects token generation speed—how responsive the model feels. 

Aim for ≥600 GB/s to keep conversations fluid.

#### Quantization

>Quantization reduces weight precision (e.g., FP16 → INT8/INT4), shrinking the VRAM footprint and allowing models 2–4× larger to run on the same hardware

![[Pasted image 20250925100141.png]]


![[Pasted image 20250925101546.png]]
### GPUs

| GPU              | Price   | VRAM          | Memory<br>Bandwidth | Slot<br>Width |
| ---------------- | :------ | :------------ | :------------------ | :------------ |
| H200 NVL         | $32,000 | 41GB<br>HBM3e | 4.8 TB/s            | 2             |
| H100 NVL         | $25,000 | 94 GB HBM3    | 3.9 TB/s            | 2             |
| A100             | $16,000 | 80GB HBM2e    | 1,935 GB/s          | 2             |
| RTX Pro 6000 BSE | $9,800  | 96GB GDDR7    | 1.8 TB/s            | 2             |
| L40s             | $7,500  | 48 GB GDDR6   | 864 GB/s            | 2             |
| L40              | $6,000  | 48 GB GDDR6   | 864 GB/s            | 2             |
|                  |         |               |                     |               |
| RTX 5090         | $2,800  | 32GB          | 1,792GB/s           | 3             |
| RTX 4090         | $2,900  | 24GB          | 1008GB/s            | 3             |
| RTX 3090         | $1,800  | 24GB          | 936 GB/s            | 3             |

### DELL PowerEdge Servers:

#### Rack Servers:

| Server | H200 | H100 | RTX Pro 6000 | L40s/L40 | Approximate Price                                                                                                                   | Max_DW_GPU_Nums |
| ------ | ---- | ---- | ------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| XE7745 | Y    | Y    | Y            | Y        | [Contact Sales](https://www.dell.com/nl-nl/shop/ipovw/poweredge-xe7745?search_redirect=xe7745)                                      | 4               |
| XE7740 | Y    | Y    | Y            | N        | [Contact Sales](https://www.dell.com/nl-nl/shop/ipovw/poweredge-xe7740?search_redirect=xe7740)                                      | 4               |
| R760xa | N    | Y    | N            | Y        | [Contact Sales](https://www.dell.com/nl-nl/shop/cty/pdp/spd/poweredge-r760xa/emea_r760xa)                                           | 4               |
| R960   | N    | N    | N            | Y        | [Contact Sales](https://www.dell.com/nl-nl/shop/servers-opslag-en-netwerken/poweredge-r960-rackserver/spd/poweredge-r960/emea_r960) | 4               |
| R7715  | N    | Y    | N            | Y        | $ 9,240                                                                                                                             | 3               |
| R570   | N    | Y    | N            | Y        | $ 6,759                                                                                                                             | 3               |
| R7615  | N    | Y    | N            | Y        | $ 5,000                                                                                                                             | 3               |
| R7525  | N    | N    | N            | Y        | $ 8,900                                                                                                                             | 3               |
| R770   | Y    | Y    | Y            | N        | $ 12,800                                                                                                                            | 2               |
| R7725  | N    | Y    | Y            | Y        | $ 11,000                                                                                                                            | 2               |
| R7625  | N    | Y    | N            | Y        | $ 6,560                                                                                                                             | 2               |
| R760   | N    | N    | N            | Y        | $ 5,850                                                                                                                             | 2               |
| XR7620 | N    | N    | N            | Y        | $ 12,380                                                                                                                            | 2               |
| R750   | N    | N    | N            | Y        | $ 4,810                                                                                                                             | 2               |
#### Tower Servers:

|      | H200 | H100 | RTX Pro 6000 | L40s/L40 | Approximate Price | Max_DW_GPU_Nums |
| ---- | ---- | ---- | ------------ | -------- | ----------------- | --------------- |
| T640 | N    | N    | N            | Y        | NA                | 4               |
| T560 | N    | N    | N            | Y        | $ 3,180           | 2               |
| T550 | N    | N    | N            | Y        | $ 2,820           | 2               |

---
## Solutions

^8e2aaa

### Experimental Solution ($3,800 ~ $5,200)

![[Pasted image 20250929140151.png]]
###### **Ryzen Quad GPU Build: Best for Inference and General Use - $3827**
This more budget-friendly build is best suited for inference, which is the process of using a pre-trained AI model to make predictions. This includes tasks like:

- Running local AI chatbots and language models: For models that can fit within the VRAM of the GPUs, this setup is very effective.
- General purpose workstation tasks: The Ryzen 9600x is a capable CPU for a variety of tasks, making this a versatile machine.

| Component       | Part                    | Price   |
| :-------------- | :---------------------- | :------ |
| **CPU**         | R5 9600x                | $197.00 |
| **Motherboard** | Gigabyte B650 AX Eagle  | $158.00 |
| **RAM**         | G.SKILL Ripjaws S5 64GB | $164.00 |
| **Cooler**      | Peerless CPU Cooler     | $35.00  |
| **Risers**      | 2x PCIe3 Risers         | $40.00  |

Key takeaway: It is a powerful and versatile machine that excels at running AI models that are already trained. It is a cost-effective solution for users who want to experiment with and use local AI without needing to train models from scratch.

###### **EPYC Quad GPU Build: Best for AI Model Training and Fine-Tuning - $5136**
The more expensive and powerful EPYC build is designed for more demanding AI workloads, specifically training and fine-tuning models. This is due to several key advantages:

- Superior PCIe Bandwidth: The EPYC platform allows all four GPUs to run at their full PCIe Gen4 x16 bandwidth. This is crucial for the massive amounts of data that need to be moved between the GPUs and system memory during training.
- Massive RAM Capacity: With support for up to 512GB of ECC RAM, this build can handle extremely large datasets and models that would be impossible to work with on the Ryzen system.
- Server-Grade Reliability: EPYC CPUs and motherboards are designed for 24/7 operation and stability, which is essential for long training runs that can take days or even weeks.

Key takeaway: This build is the ideal choice for developers, researchers, and enthusiasts who are serious about creating and customizing their own AI models. 

| Component       | Part                | Price   |
| :-------------- | :------------------ | :------ |
| **CPU**         | AMD EPYC 7702       | $450.00 |
| **Motherboard** | MZ32-AR0            | $558.00 |
| **RAM**         | DDR4 2400 ECC 512GB | $575.00 |
| **Cooler**      | Corsair H170i       | $160.00 |
| **Risers**      | 4x PCIe4 Risers     | $160.00 |
###### **Shared Components (Used in Both Set-ups)**

| Component     | Part                     | Price     |
| :------------ | :----------------------- | :-------- |
| **Chassis**   | GPU Rack Frame           | $55.00    |
| **GPU**       | 4 x 3090 24GB            | $2,800.00 |
| **PSU**       | CORSAIR HX1500i          | $300.00   |
| **NVMe/SSD**  | NVMe Gen4 1TB            | $70.00    |
| **Accessory** | HDD Rack Screws for Fans | $8.00     |

### Industrial Solution(More than $52,000)

![[Pasted image 20250929142208.png]]

| Component | Part             | Price   | Source                                                                                                                                                                     |
| --------- | ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Server    | PowerEdge R760xa | $25,000 | https://www.dell.com/nl-nl/shop/cty/pdp/spd/poweredge-r760xa/emea_r760xa                                                                                                   |
| GPU       | H100 NVL         | $27,000 | https://www.etb-tech.com/dell-nvidia-h100-nvl-graphics-accelerator-94gb-full-height-bracket-vid00102.html?srsltid=AfmBOooUvF2hQ74pH6zW51qviyyzsMQn61Npn3KuF9G40bLPKbgRJuPv |
>Other components are not calculated yet
## Other information

^fd7fe0

###  A VRAM Showdown Between Fine-Tuning and Inference for a 30B Local LLM

^a4ad18

Running a 30 billion parameter Large Language Model (LLM) locally is a significant undertaking, but the hardware requirements diverge dramatically depending on whether you are fine-tuning the model or simply running it for inference. Fine-tuning, the process of adapting a pre-trained model to a specific task, is substantially more memory-intensive than inference, which involves generating text from a ready-to-use model.

This comparison breaks down the Video RAM (VRAM) requirements for both processes, using a 30B parameter model as the benchmark.

#### Fine-Tuning a 30B LLM

Fine-tuning a 30B model involves not just loading the model's parameters into VRAM, but also storing optimizer states, gradients, and activations, which drastically increases memory consumption. Here’s how different strategies stack up.

**Background:**
*   A 30B parameter model, when loaded in the standard **FP16 (half-precision floating-point)** format, requires a baseline of `30B * 2 bytes/parameter = 60 GB` of VRAM just to store the model weights.

---

1. FULL (Full Fine-tuning) for a 30B Model
	*   **Operation:** This method loads the entire 30B parameter model and allows every single parameter to be updated during training.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters:** 60 GB (FP16)
	    *   **Optimizer States:** Using a standard optimizer like AdamW typically requires storing multiple states for each parameter. These are often in a higher precision (FP32), leading to `30B * 4 bytes/parameter = 120 GB`.
	    *   **Gradients:** Gradients, the same size as the model parameters, must be stored, adding another `30B * 2 bytes/parameter = 60 GB`.
	    *   **Activations:** Intermediate values from the forward pass are stored for backpropagation. The memory for this is dependent on batch size and sequence length but can easily add tens of gigabytes.
	    *   **Total Estimate:** `60 GB (parameters) + 120 GB (optimizer) + 60 GB (gradients) + tens of GB (activations) = at least 240 GB` or more.
	
	*   **Conclusion:**
	    *   Full fine-tuning of a 30B model is a monumental task, far beyond the reach of any single consumer or even most professional GPUs. It necessitates a powerful multi-GPU setup, such as **four 80GB A100/H100 GPUs**, to even begin.
	    
2. LoRA (Low-Rank Adaptation) for a 30B Model
	*   **Operation:** LoRA freezes the 60GB of original model weights and injects small, trainable "adapter" matrices into the model. Only these adapters, which have a tiny fraction of the original parameters, are updated.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters:** 60 GB (FP16, frozen)
	    *   **LoRA Adapter Parameters & Optimizer:** The adapters and their corresponding optimizer states are very small, typically only consuming a few gigabytes of VRAM.
	    *   **Activations:** Memory for activations is still required, but techniques like gradient checkpointing can help reduce this footprint.
	    *   **Total Estimate:** `60 GB (frozen parameters) + a few GB (LoRA components and activations) = approximately 60-70 GB`.
	
	*   **Conclusion:**
	    *   LoRA dramatically reduces the VRAM overhead of fine-tuning, making it feasible on a **single 80GB A100/H100 GPU**.
	    *   However, it remains out of reach for standard consumer GPUs like the 24GB RTX 4090, as the base model alone exceeds the available VRAM.
	    
3. QLoRA (Quantized Low-Rank Adaptation) for a 30B Model
	*   **Operation:** QLoRA takes efficiency a step further. It first quantizes the 30B model down to 4-bit precision before loading it, then freezes these quantized weights and trains small LoRA adapters on top.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters (Quantized):** The 30B parameters are loaded in 4-bit, drastically reducing their size to `30B * 0.5 bytes/parameter = 15 GB`.
	    *   **LoRA Adapter Parameters & Optimizer:** These remain small, consuming a few gigabytes in a higher precision like BF16.
	    *   **Activations & Optimizations:** QLoRA employs clever techniques like paged optimizers and double quantization to manage the memory from activations and optimizer states, keeping the total footprint low.
	    *   **Total Estimate:** `15 GB (4-bit parameters) + a few GB (LoRA components and optimized overhead) = approximately 20-30 GB`.
	
	*   **Conclusion:**
	    *   QLoRA is a game-changer for accessibility. It brings the VRAM requirement for fine-tuning a 30B model down to a level that is manageable for high-end consumer hardware.
	    *   A single **24GB VRAM GPU (e.g., NVIDIA RTX 3090/4090)** can successfully fine-tune a 30B model using QLoRA, making advanced local LLM customization accessible to a much wider audience.

#### Inference with a 30B LLM

Inference is a far less demanding task as it only requires the model weights and the activations for the current input (KV cache) to be held in VRAM. There are no gradients or optimizer states to store.

1. FP16 (Half-Precision) Inference
	*   **Operation:** Loads the full 30B parameter model in 16-bit precision to generate text.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters:** 60 GB (`30B * 2 bytes`)
	    *   **KV Cache & Overhead:** Additional memory is needed for the key-value cache (which stores attention information for the generated sequence) and other minor overheads. This can add several gigabytes depending on the context length.
	    *   **Total Estimate:** `~60-65 GB`.
	
	*   **Conclusion:**
	    *   Running a 30B model at full half-precision requires a high-end professional GPU like an **80GB A100/H100** or a dual-GPU setup with consumer cards.
	    
2. INT8 (8-bit Quantized) Inference
	*   **Operation:** The model's weights are quantized to 8-bit integers, halving the memory footprint.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters:** 30 GB (`30B * 1 byte`)
	    *   **KV Cache & Overhead:** A few gigabytes for context and overhead.
	    *   **Total Estimate:** `~30-35 GB`.
	
	*   **Conclusion:**
	    *   This level of quantization makes 30B model inference possible on professional cards with around 40-48GB of VRAM, or potentially a dual-GPU consumer setup. It offers a good balance between performance and resource usage with minimal quality loss.
	    
3. 4-bit Quantized Inference (e.g., GPTQ, GGUF)
	*   **Operation:** The model is aggressively quantized down to 4-bits per parameter. This is the most common method for running large models on consumer hardware.
	*   **VRAM Requirement Estimate:**
	    *   **Model Parameters:** 15 GB (`30B * 0.5 bytes`)
	    *   **KV Cache & Overhead:** A few gigabytes for context, with some implementations allowing the KV cache to be offloaded to system RAM.
	    *   **Total Estimate:** `~17-24 GB`.
	
	*   **Conclusion:**
	    *   4-bit quantization makes running a 30B model highly accessible. A single **24GB VRAM consumer GPU (NVIDIA RTX 3090/4090)** is sufficient to run these models effectively, with some variants fitting comfortably within 20GB of VRAM.

#### Summary Table: Fine-Tuning vs. Inference

| **Process**     | **Method** | **Core Operation**                                                      | **VRAM (Approx.)** | **Single-Card Hardware (Rough)** |
| :-------------- | :--------- | :---------------------------------------------------------------------- | :----------------- | :------------------------------- |
| **Fine-Tuning** | **FULL**   | Update all parameters; model, optimizer, activations in full precision. | **240 GB+**        | Multiple 80GB A100/H100s         |
|                 | **LoRA**   | Freeze original model (FP16), train small LoRA adapters.                | **60-70 GB**       | Single 80GB A100/H100            |
|                 | **QLoRA**  | Quantize model to 4-bit and freeze, train small LoRA adapters.          | **20-30 GB**       | Single 24GB RTX 3090/4090        |
| **Inference**   | **FP16**   | Load full 16-bit model for text generation.                             | **60-65 GB**       | Single 80GB A100/H100            |
|                 | **INT8**   | Load 8-bit quantized model for text generation.                         | **30-35 GB**       | Single 48GB Professional GPU     |
|                 | **4-bit**  | Load 4-bit quantized model for text generation.                         | **17-24 GB**       | Single 24GB RTX 3090/4090        |
## References

^4a397c

| **LLM**                                                                                                                                                                                                                                                                                                                                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MCPVerse: An Expansive, Real-World Benchmark for Agentic Tool Use](https://arxiv.org/html/2508.16260v1)                                                                                                                                                                                                                                                                 |
| [UNSLOTH Quantized Models](https://docs.unsloth.ai/models/deepseek-v3.1-how-to-run-locally)                                                                                                                                                                                                                                                                              |
| [Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](https://arxiv.org/html/2505.02390v1)                                                                                                                                                                                                                                                          |
| [Live LLM Benchmark](https://livebench.ai/#/?organization=Alibaba%2CDeepSeek%2CGoogle%2CMistral+AI%2CMeta)                                                                                                                                                                                                                                                               |
| [Model Comparison - Gemini 2.5 Flash (Reasoning)](https://artificialanalysis.ai/models/gemini-2-5-flash-reasoning?models=gpt-oss-120b%2Cgpt-oss-20b%2Cgemini-2-5-flash-reasoning%2Cgemini-2-5-pro%2Cmagistral-small-2509%2Cdeepseek-v3-1%2Cdeepseek-v3-1-reasoning%2Cdeepseek-r1%2Cqwen3-30b-a3b-2507-reasoning%2Cqwen3-30b-a3b-2507%2Cqwen3-4b-2507-instruct-reasoning) |
| [5 Essential LLM Quantization Techniques Explained](https://apxml.com/zh/posts/llm-quantization-techniques-explained)                                                                                                                                                                                                                                                    |
| [LLMs quantization naming explained](https://andreshat.medium.com/llm-quantization-naming-explained-bedde33f7192)                                                                                                                                                                                                                                                        |
| [Fine-tune and run Qwen3-2507](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune/qwen3-2507)                                                                                                                                                                                                                                                                 |
| [DeepSeek R1 Hardware Requirements Explained](https://www.youtube.com/watch?v=5RhPZgDoglE)                                                                                                                                                                                                                                                                               |
| [Mixture-of-Experts (MoE) LLMs: The Future of Efficient AI Models](https://www.youtube.com/watch?v=xGaA0qLdCIY&t=109s)                                                                                                                                                                                                                                                   |
| [Estimating Memory Requirements for Large Language Models Based on User Load and Model Precision](https://www.linkedin.com/pulse/estimating-memory-requirements-large-language-models-based-tripathi-ozd3c)                                                                                                                                                              |
| [What kind of specs to run local llm and serve to say up to 20-50 users](https://www.reddit.com/r/LocalLLaMA/comments/185a3mh/what_kind_of_specs_to_run_local_llm_and_serve_to/)                                                                                                                                                                                         |
| [Struggling on local multi-user inference? Llama.cpp GGUF vs VLLM AWQ/GPTQ.](https://www.reddit.com/r/LocalLLaMA/comments/1lafihl/struggling_on_local_multiuser_inference_llamacpp/)                                                                                                                                                                                     |
| [Does quantization harm results?](https://www.reddit.com/r/LocalLLaMA/comments/154tsux/does_quantization_harm_results/)                                                                                                                                                                                                                                                  |
| **GPUs**                                                                                                                                                                                                                                                                                                                                                                 |
| [L40S vs. A100 vs. H100 vs. B200: which GPU for your workload?](https://modal.com/blog/nvidia-l40s-price-article)                                                                                                                                                                                                                                                        |
| [Set up suggestion: H200 vs RTX Pro 6000](https://www.reddit.com/r/LocalLLaMA/comments/1kwn7t4/setup_recommendation_for_university_h200_vs_rtx/)                                                                                                                                                                                                                         |
| [Nvidia L40s](https://www.reddit.com/r/LocalLLaMA/comments/1gqo5ox/nvidia_l40s/)                                                                                                                                                                                                                                                                                         |
| [RTX 4090 vs L40S for Server](https://www.reddit.com/r/MachineLearning/comments/1elenk8/d_rtx_4090_vs_l40s_for_server/)                                                                                                                                                                                                                                                  |
| [3090 vs A100](https://www.reddit.com/r/LocalLLaMA/comments/1476pco/better_inference_on_3090_than_a100/)                                                                                                                                                                                                                                                                 |
| [4090 vs 3090](https://youtu.be/xzwb94eJ-EE?si=NmPZzE5v9e9Oxl_Y)                                                                                                                                                                                                                                                                                                         |
| [Price Website (Tweakers)](https://tweakers.net)                                                                                                                                                                                                                                                                                                                         |
| **DELL Servers**                                                                                                                                                                                                                                                                                                                                                         |
| [Dell Server GPU Matrix](https://www.delltechnologies.com/asset/en-iq/products/servers/briefs-summaries/poweredge-server-gpu-matrix.pdf)                                                                                                                                                                                                                                 |
| [Dell Tower Servers - Quick Ref](https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-tower-quick-reference-guide.pdf)                                                                                                                                                                                                              |
| [Dell Rack Servers - Quick Ref](https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-rack-quick-reference-guide.pdf)                                                                                                                                                                                                                |
| [Dell T640 for a 4x 3090 build](https://www.reddit.com/r/LocalLLaMA/comments/1j09qk1/dell_t640_for_a_4x_3090_build/)                                                                                                                                                                                                                                                     |
| **Guide**                                                                                                                                                                                                                                                                                                                                                                |
| [LLM Inference: VRAM & Performance Calculator](https://apxml.com/tools/vram-calculator)                                                                                                                                                                                                                                                                                  |
| [HOW TO INSTALL DEEPSEEK-R1 LOCALLY: FULL $6K HARDWARE & SOFTWARE GUIDE](https://rasim.pro/blog/how-to-install-deepseek-r1-locally-full-6k-hardware-software-guide/)                                                                                                                                                                                                     |
| [DeepSeek-V3.1-Terminus on your local device](https://www.reddit.com/r/LocalLLM/comments/1np1o9e/you_can_now_run_deepseekv31terminus_on_your_local/)                                                                                                                                                                                                                     |
| https://digitalspaceport.com/local-ai-home-server-build-at-high-end-5000-price/                                                                                                                                                                                                                                                                                          |
