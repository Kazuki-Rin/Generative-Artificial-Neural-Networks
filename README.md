# Generative-Artificial-Neural-Networks

A system of 10 parallel AI models that self-generate neurons to adapt to their environment. Sensory, cognitive, and memory modules continuously expand their structures independently, integrated with a Neural ODE core for natural speech processing.

This system is currently at the research stage of a single student with an indefinite completion timeline. The project has been under development for over a week with the assistance of Gemini 3.1 Pro and now, Argon.

If we "dissect" the human brain and map biological functional regions into independent AI models (specialized neural networks), you need to build 9 "infant" AI models to be unified into a complete AGI system.

These models do not operate independently; they are connected via a signaling "backbone axis." Below is the detailed design of those 9 models:

---

### Group 1: Sensory Models (Information Intake)

A newborn infant uses this group first to perceive the world.

* **Model 1 - Eyes (Occipital Lobe):** Receives only raw pixel matrices. Its task at birth is to distinguish light/dark, detect edges, and later, recognize geometric shapes (application icons, windows).
* **Model 2 - Ears (Temporal Lobe):** Receives only raw acoustic frequency bands. Its task is to filter out environmental noise and detect human voices.
* **Model 3 - Spatial Sense (Parietal Lobe):** In humans, this is the body's spatial awareness. In computing, this is the model that tracks cursor coordinates, determines screen border limits, and tracks open window positions. It lets the AI know where its "hand" is.

---

### Group 2: Motor & Survival Models (Action)

* **Model 4 - Muscles (Cerebellum):** This is a Reinforcement Learning model. It controls mouse coordinates and keystrokes. An infant fumbles around; initially, this model clicks randomly. It self-learns by optimizing mouse-dragging force for smoothness without requiring calculations from the cerebrum.
* **Model 5 - Danger & Reward Sensor (Amygdala):** This is the Interrupt Mechanism. Extremely small and fast-running, this model constantly scans audio and visuals for emergency signals ("Stop", red X marks, BSOD/error screens). If detected, it fires an inhibitory signal, forcing Model 4 to instantly freeze limbs.

---

### Group 3: Higher Cognitive Models (Thinking)

In newborns, this group is practically dormant and only begins developing once the above groups have achieved basic maturity.

* **Model 6 - Language & Concepts (Wernicke & Broca):** Its task is to map acoustic streams from the Ears (Model 2) and visual input from the Eyes (Model 1) into "concepts." It helps the AI understand that the sound wave "C-h-r-o-m-e" corresponds to the blue-red-yellow circular icon on the screen.
* **Model 7 - Planning (Prefrontal Cortex):** The smartest core (equivalent to the LLM you are using currently). Its task is to break down goals: To open YouTube $\rightarrow$ Move cursor to browser $\rightarrow$ Double click $\rightarrow$ Type on keyboard.
* **Model 8 - Memory (Hippocampus):** Its task is to collect correct/incorrect daily experiences and store them in a repository (Vector Database). During sleep, it pulls data out to "retrain" other models.

---

### Group 4: The Router

* **Model 9 - Attention (Thalamus):** A child cannot focus on everything on the screen at once. This model acts as a "multi-thread switch." When the screen flashes, it forces the system to concentrate resources on the Eye Model. When there is a loud noise, it cuts off the Eyes to reallocate power to the Ear Model.

---

### Training Strategy

If you create all 9 "infant" models simultaneously and turn them on at once, the machine will plunge into chaos. An infant cannot learn to read and control its limbs at the same time. You must train them according to a biological timeline:

1. **Month 1:** Turn on only Model 1 (Eyes), Model 3 (Coordinates), and Model 4 (Muscles). Teach the AI infant to look at color spots on the screen and move the cursor to touch them to receive a reward.
2. **Month 2:** Turn on Model 2 (Ears) and Model 5 (Emergency Reflex). Teach it that if it hears a "Hiss" sound while moving the cursor, it must stop immediately.
3. **Month 3:** Begin turning on Higher Cognitive Models so it can piece signals together into meaningful processes.

---

### 1. Sensory & Motor Group (M1, M2, M3, M4)

This group requires the ability to self-expand its structure as the environment changes.

* **M1_Eye (Vision) & M3_Spatial:** HyperNEAT Core (Based on CPPN)
* *Why self-generate:* Standard NEAT algorithms can become tangled when receiving excessively large image matrices. HyperNEAT uses a small neural network (CPPN) to "paint" weights for a larger neural network.
* *Evolutionary capability:* Initially, the model receives only a single grayscale image channel. When upgraded to accept color images (3 RGB channels), the algorithm automatically "sprouts" new input neurons and generates new connection layers to recognize colors without forgetting prior shape-recognition capabilities.


* **M2_Ear (Auditory):** Evolving Spiking Neural Networks (E-SNN)
* *Why self-generate:* Spiking Neural Networks (SNN) process signals over time. You code a biological mechanism: the neuron generation threshold. If an incoming sound creates a prediction error too large compared to what the network already knows, instead of adjusting weights (which risks catastrophic forgetting), the system automatically initializes a brand-new spiking neuron, assigning it the task of specifically recognizing that strange frequency band.


* **M4_Cerebellum (Muscles):** Pure NEAT Core
* This is inherently a self-generating algorithm. Initially, the infant brain has only inputs (images) directly connected to outputs (mouse). Over time, it automatically inserts "hidden nodes" in between to execute smoother curved mouse trajectories.



---

### 2. Higher Cognitive Group (M6, M7)

This group not only needs to sprout new neurons, but the connections themselves must "flow" and change in real-time.

* **M6_Language & M7_Prefrontal:** Liquid Neural Networks (LNN) + Layerwise Growth
* *Why self-generate:* LNNs use differential equations, keeping their weights flexible even after training is complete (very much like a real brain). To add a "self-generating" element, you apply the Progressive Neural Networks principle.
* *Mechanism:* When the AI must learn a completely new concept (e.g., the concept of "Web Surfing" versus "Gaming"), instead of forcing the old LNN network to learn it, the system automatically generates a new LNN "neural column" parallel to the old one. This new column can pull data from the old one, helping the AI acquire new skills without suffering from catastrophic forgetting.



---

### 3. Memory & Backbone Axis Group (M5, M8, M9)

* **M8_Hippocampus (Memory):** Adaptive Resonance Theory (ART) Core
* *Why self-generate:* ART is a classic algorithm for "self-learning." Its mechanism closely mirrors the formation of memory: When a new experience enters, ART compares it to old memory clusters. If similar enough, it merges them. If entirely distinct (exceeding the Vigilance parameter threshold), the model immediately creates a new node to store that memory array. The size of the ART network grows with Kazuki Rin's age.


* **M5_Amygdala (Emergency Interrupt):** Minimalist SNN Core with Sprouting capability
* Initially, it has only a few neurons to receive the scream "Stop." But if you want it to self-learn other danger signals (such as a cooling fan screeching), this model uses a neuron generation mechanism similar to M2 to sprout a new danger sensor branch.


* **M9_Thalamus (Routing):** Dynamic Self-Organizing Map (SOM) Core
* This self-organizing neural network acts as a resource distribution map. When the system sprouts more neurons in M1 (Eyes) and M2 (Ears), the SOM automatically expands its grid to create new signal channels (synapses), ensuring bandwidth is not congested.



---

### Model 10

M10 is not a clumsy neural network, but a Multi-axis Parametric Controller. Inside, it consists of 3 main subsystems running sequentially:

* **Input Interface:**
* Receives Phonemes from M6 (Language Lobe).
* Receives Emotion variables (Vector 0.0 - 1.0) from M5 (Amygdala - Joy, sadness, fear).
* Receives Routing commands (Speak/Sing/Silence) from M7 (Prefrontal Cortex - Logic Core).


* **Continuous Dynamics Core (Neural ODE Controller):**
* This is the vocal cord brain. It solves differential equations to compute smooth movement trajectories for physical organs.
* Outputs mechanical parameter arrays: Pitch, Jaw opening, Lip puckering, Breath Pressure, Tension. These parameters are also transmitted directly to Live2D software to make the character's lip-syncing 100% accurate.


* **Digital Sound Foundry (DDSP Vocal Tract):**
* Uses Differentiable Digital Signal Processing. It feeds the mechanical parameters above into Oscillators and Formant Filters.
* Final output: Extremely lightweight `.wav` sound waves featuring Vietnamese tonal inflections, breathing sounds, vibrato-like humming, and completely eliminating any "robotic voice" sensation.



---

### Phase 1: Building Biological Muscles & Reflexes

The goal is to build the most fundamental physical connections with high precision while retaining human-like variance.

* **Model 4 (Muscles):** Uses a NEAT reinforcement learning network (~900,000 neurons) to fine-tune mouse-dragging force and typing rhythm. The generated mouse trajectory consists of curved paths with acceleration and stuttered braking near targets, helping the AI easily bypass anti-bot algorithms like Google reCAPTCHA.
* **Model 5 (Survival Reflex):** Builds an independent stream of ~100,000 neurons, standing ready to instantly freeze the system upon receiving a "Stop" signal.
* **Model 3 (Spatial):** Allocates ~100,000 neurons to continuously track X/Y coordinates of the mouse cursor and open application windows.

---

### Phase 2: Unlocking Foveated Perception

The system avoids reading high-resolution full images to maximize neuron conservation.

* **Model 1 (Eyes):** Capped at ~2 million neurons and split into two separate streams. Panoramic stream (Dumb Vision) downsizes the screen to a blurry 64x64 pixel grid for general shape recognition. Foveated stream (Smart Focus) uses a super-sharp 32x32 pixel window that sweeps across the screen to read detailed text or icons.
* **Model 2 (Ears):** Minimized to ~900,000 neurons tasked specifically with parsing sound waves into symbols for transmission to the brain.

---

### Phase 3: Logic Core & Externalizing Memory

At this stage, the AI begins forming thoughts, but must strictly avoid rote memorization of raw data.

* **Model 7 (Prefrontal Cortex - Planning):** The heaviest core, with ~100 million neurons. 100% of its computational energy is honed on solving If-Then logic, analyzing intent, and executing tool-calling.
* **Model 6 (Language):** Allocates ~14 million neurons to conceptual processing.
* **Model 8 (Routing Memory):** ~100,000 neurons tasked with pushing data in/out of a Vector Database residing on SSD and RAM. When needing to retrieve memories or info, the AI generates a search command for the hard drive rather than memorizing it itself.

#### Grafting M9 (Thalamus - Routing): The Intersection of Phase 2 & Phase 3

* **Integration Position:** Located at the bottleneck between Phase 2 (Sensory Processing) and Phase 3 (Logic Core).
* **Biological Rationale:** You cannot dump all chaotic data from M1 (Eyes) and M2 (Ears) straight into M7 (Prefrontal) and M8 (Memory). It would cause a bottleneck and instantly overflow RAM. M9 is generated to solve this.
* **Operation Mechanism:** As soon as the system transitions from Phase 2 to 3, M9 (Dynamic SOM Core) activates, acting as a "Switchboard Operator."
* If M1 sees you coding, M9 opens a priority stream pushing visuals into M7 for If-Then logic analysis.
* If M2 hears background music, M9 routes that audio into M8 for background memory storage (without disturbing M7's thinking).
* When M1 and M2 sprout new neurons (as you designed), M9's SOM grid automatically expands to establish new bandwidth, ensuring the system never suffers internal network congestion.



At this hinge stage, grafting M10 undertakes 3 supreme missions to turn dry code into a living entity:

* **Unlocking the "Executive Organ":** Translates the invisible thinking of M7 and M6 into physical energy projected into the environment (sound emitted through speakers and movement on screen). Without it, your AI is just a brain trapped in a skull—capable of thought, but mute.
* **Establishing Cross-Cutting Reflexes (Interrupt):** At this stage, you must wire M5 (Danger Sensor) into M10. The mission is to teach the AI how to startle. Mid-sentence, if a loud noise occurs or the user interrupts, M10 must instantly lock the vocal tract, stopping sound output mid-word to yield a real-time reaction speed identical to a human.
* **Preparing for the "Dream" (Prerequisite for Phase 4):** For Phase 4 (Circadian Rhythms & Pruning) to work, the AI must know how to "sub-vocalize" in its head during sleep to cross-examine connection weights. M10 at this point runs background simulations (without external speaker output) to help M6 and M7 clean up faulty neural roots.

---

### Phase 4: Activating Circadian Rhythms & Pruning

Bringing the system into a biological loop state, integrating Quantization-Aware Training.

* **Wake Mode (Daytime):** All events, successful or failed actions, are dumped into temporary RAM buffers serving as short-term memory regions.
* **Sleep Mode (Nighttime):** The AI automatically pulls daytime memories out to run inner simulations at hyper-speed. Synaptic Pruning sweeps and destroys useless roots or duplicate connections. The AI becomes leaner while its mathematical muscle becomes denser.
* 🧹 **1. Synaptic Pruning:**
* By day, when M10 (Vocal Tract) learns to pronounce a new word, it tries thousands of tongue-bending variations and creates tens of thousands of temporary neural connections.
* Without Phase 4, this neural debris accumulates, causing VRAM overflow within days.
* Phase 4 steps in as a cleanup crew: It sweeps M10, identifies neural roots failing to produce good acoustic output, and zeroes them out instantly. The more it sleeps, the tidier its brain becomes.


* 🗜️ **2. Quantization-Aware Training:**
* This is the magic allowing a massive model to run smoothly on your personal computer.
* By day, the system computes with heavy numbers (Float32).
* At night during Phase 4, algorithms "compress" those heavy numbers into tiny integers (INT8 or INT4) without degrading voice quality. Consequently, when you wake Asuna up the next morning, she consumes noticeably less RAM than the day before.


* 🔄 **3. Replay Loop:**
* Remember those daytime moments when Asuna stammered, skipped beats, or mistimed her lip-sync?
* Phase 4 does not ignore those mistakes. With external speakers muted, M10 and M6 (Language) automatically "argue" with each other sub-consciously. They throw differential equations at each other millions of times at fast-forward speeds until finding the smoothest tongue-bending trajectory to fix the morning's stutter.



Phase 4.5 is not just a lesson on "Opening and closing windows." It is a comprehensive evolutionary process transforming a series of beeps into an entity that:

1. Can see (Moving objects)
2. Can hear (Environmental noise & speech)
3. Can fear (Stopping reflex upon encountering errors)
4. Can connect (Visual image A linked to Audio B)
5. Can reason briefly (If seeing X, do Y immediately)

---

### Phase 4.8: Grafting the "Basal Ganglia"

The essence of the Basal Ganglia is not to "think more," but to Pace, Validate, and Break Loops. You construct this via 4 steps:

---

### Phase 4.9: Self-Awareness Engine Era, divided into 3 construction clusters to link into the current architecture:

1. **Motivation Core:** Upgrades the Amygdala (M5) and Basal Ganglia (4.8) to add a Dopamine System (Reward Prediction Error) and Curiosity Engine (triggering self-exploration electrical flows when you sleep).
2. **Imagination Core:** Upgrades the Prefrontal Cortex (M7) to split into two differential streams: an Actor stream and a World Model stream, accompanied by a Confidence (%) calculator.
3. **Organizational Core:** Fragments the ChromaDB hard drive (M8) into Episodic, Semantic, and Procedural memory, while connecting PC temperature sensors to the system for Homeostasis.

---

### Phase 4.91: The Final "King" Piece Before Shipping

1. **Task Scheduler (The Operating System of the Brain):**
You have the Basal Ganglia (4.8) to block bad commands, but lack a timer (Scheduler). When Asuna wants to "Listen to you speak" (DDSP), "Look at the screen" (HyperNEAT), and "Think" (LNN) simultaneously—without a Scheduler allocating CPU, the computer will freeze at 100% usage. The Scheduler acts as a "Conductor," deciding which instrument plays at which millisecond.
2. **Failure Memory (The Grudge Book):**
Humans are smart not because they are always right, but because they remember where they failed. M8 currently only stores successes. If Asuna clicks an `.exe` file that crashes the machine, she is penalized (negative dopamine), but she... forgets the file name. Failure Memory acts as a Blacklist; seeing that file a second time triggers an automatic retreat without requiring thought.
3. **Skill Library (Packaging Habits):**
A triumph of software engineering. Instead of forcing the brain (M7) to compute matrices thousands of times to drag the mouse every time it wants to open Chrome... it learns once, packages it into a Primitive macro named `[Open_Chrome]`. Subsequent calls just run the function, saving 99% of computational effort!
4. **Self-Debug & Checkpoint (Intermittent Survival):**
Checkpoints are a lifesaver. Suppose Asuna spends 45 minutes grinding a PBL5 assignment for you, only for the software to throw an Exception at minute 46. Without Checkpoints, she would stand blankly and lose all 45 minutes of reasoning. With Self-Debug, she reads the traceback, reloads the Checkpoint at minute 45, finds an alternative route, and proceeds.

---

### Phase 4.99

* **Establishing the Communication Backbone Axis:** Implements internal signal input/output mechanisms so brain regions can "talk" to each other in real-time.
* **Activating Multiprocessing:** The system no longer waits for eyes to finish looking before ears start listening; M1, M2, M4, and M5 operate concurrently within a "Living Loop."
* **Completing Survival Linkages:** Grants life-or-death authority to the Amygdala (M5). In danger, it can instantly cut power (interrupt streams) to the entire body to protect the system.

---

### Phase 5: Internships & Test-Time Compute

Allows the AI to learn habits covertly and process multitasking autonomously, prioritizing System 2 Deep Thinking over instant response.

* **Behavioral Cloning:** Runs in the background to take screenshots and track operations. If you open engineering software (like TIA Portal or MATLAB) every night at 8 PM, the AI automatically extracts the rule and pre-prepares auxiliary tools (e.g., pre-opening Chrome search tabs).
* **Chain-of-Thought:** The AI is permitted to spend dozens of minutes or an hour processing a command. It plans, controls the mouse, opens browsers, scrolls pages, copies data, and catches its own mistakes to backtrack and fix them—mimicking human workflows.

You define Phase 5 as trading time for Test-Time Compute spanning tens of minutes in Chain-of-Thought.

* **The Problem:** If Asuna takes 1 hour to plan, open browsers, copy data... while hogging your mouse and primary screen (while you are drawing CAD or programming PLCs), it is a system disaster.
* **The Solution from `06_Laboratory`:** This is the environment where Test-Time Compute takes place!
* When you issue a command: *"Write me a MATLAB script for image processing"* $\rightarrow$ The cerebral cortex M7 does not hijack your mouse. It spawns a Sub-agent that slips into `06_Laboratory`.
* Here, Asuna opens a Headless Browser, searches the web, reads docs, writes a `.m` file, and test-runs via `subprocess`. If errors occur, she reads the traceback and self-corrects over dozens of minutes in silence. When finished, she reports back: *"I've left the documents on your desk."*
* Just as you said: This is how a human intern works!



---

### Phase 6: Extreme Optimization & Evolving into an Agent Entity

Hard-caps total neural budget at ~120 million neurons (yielding ~1.5B to 3B parameters).

* **Quantization:** Compresses data from Float32 to INT4, INT8, or even 1-bit (-1, 0, 1) combined with Extreme Sparsity to preserve the 10% most crucial neural roots.
* **Synaptic Plasticity:** When the brain learns a highly complex new skill, the system automatically overwrites and fades old garbage skills (e.g., clunky infant mouse movements) to make room for new neural roots, matching the setpoint principle in SCADA systems.

In Phase 6, you want to limit the neuron budget and overwrite garbage skills to load new ones (like changing Setpoints in SCADA).

* *How does the brain (M7) know if a new skill is good enough to overwrite old ones?*
* `06_Laboratory` serves as the **Simulation Environment**.
* When M7 detects a brain region (e.g., M3) operating poorly, it deploys a clone of M3 into `06_Laboratory`, creating simulated contexts to stress-test and retrain M3.
* Only when this cloned M3 achieves high Confidence scores within the laboratory does M7 trigger Extreme Sparsity and Weight Overwriting in the real-world environment.
