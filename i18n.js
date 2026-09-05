/**
 * English copy and shared interface labels.
 * Media paths and quantitative values are inherited from content.js so the
 * two language versions always display the same evidence.
 */
(function buildLocalizedContent() {
  const zh = window.PROJECT_DATA;
  if (!zh) throw new Error("PROJECT_DATA must be loaded before i18n.js");

  const en = JSON.parse(JSON.stringify(zh));

  Object.assign(en.project, {
    kicker: "TECHNICAL REPORT · SEPTEMBER 2026",
    titleEn: "Production-Oriented Visual Dubbing",
    titleCn: "Task-adaptive post-training and two-step distillation",
    affiliation: "TaoLive AIGC · Alibaba Group",
    abstract: "Visual dubbing must synchronize mouth motion with replacement speech while preserving identity, appearance, and temporal consistency. TBDub performs production-oriented task-adaptive post-training on X-Dub's mask-free video-editing framework, using production-domain data, conditioning-side augmentation, an identity-matched oral prior, and multi-layer HuBERT features to obtain a 30-step Teacher. We then adapt DMD/DMD2 to conditional video editing and distill the Teacher into a two-step Student. On 38 TalkVid clips, the Teacher improves all reported reconstruction, perceptual, identity, and synchronization metrics over X-Dub. The Student largely retains the Teacher's visual quality and audiovisual synchronization while reaching 7.13 effective FPS on the 512×512 VAE-to-VAE generation path on a single NVIDIA H20 GPU.",
    summaryIntro: "TBDub connects production-domain post-training with task-aware few-step distillation. It targets recurring failures in livestream, studio, and AIGC-generated content and offers a deployment-oriented balance among identity and oral-detail fidelity, temporal stability, robustness, and inference efficiency.",
    demoIntro: "Qualitative results cover multilingual self-driven reconstruction, redrawing of MiniMax H3 generated videos, and cross-audio driving under large poses, occlusion, rapid head motion, and unconstrained real-world conditions. Video is more informative than isolated frames for evaluating synchronization, oral detail, and temporal stability.",
    authorNote: "* Equal contribution.  † Corresponding author.",
  });

  en.project.focusAreas = [
    {
      index: "01",
      title: "Identity & Detail Fidelity",
      titleCn: "Preserving fine facial characteristics",
      text: "Preserve lip color, tooth geometry, skin texture, and eye details while suppressing identity, color, and sharpness drift caused by local editing or long-sequence generation.",
    },
    {
      index: "02",
      title: "Long-range Temporal Stability",
      titleCn: "Consistent appearance over time",
      text: "Improve cross-frame appearance and structural consistency to reduce illumination flicker, tooth adhesion, scale oscillation, and cumulative visual degradation over long inference horizons.",
    },
    {
      index: "03",
      title: "Robust Visual Editing",
      titleCn: "Reliable performance in challenging scenes",
      text: "Maintain stable editing under occlusion, large head rotations, rapid motion, and motion blur, reducing mouth-boundary artifacts, incorrect overlays, and local generation failures.",
    },
    {
      index: "04",
      title: "Cross-driven & Cross-domain",
      titleCn: "Generalization beyond matched reconstruction",
      text: "Improve phoneme-to-mouth correspondence, oral-detail clarity, and visual integration when the driving speech differs from the original articulation or the source video is AIGC-generated.",
    },
    {
      index: "05",
      title: "Task-aware Few-step Distillation",
      titleCn: "Guidance-aware compression for deployment",
      text: "Construct a guided Teacher score from X-Dub's three-route dynamic CFG, then use conditional distribution matching to compress the 30-step Teacher into a two-step, single-route Student.",
    },
  ];

  const settingCopy = {
    self: {
      titleCn: "",
      description: "The target video is driven by its own speech. This setting primarily evaluates reconstruction fidelity, identity preservation, lip synchronization, and temporal stability.",
    },
    cross: {
      titleCn: "",
      description: "The target video is re-animated with replacement speech. This setting evaluates articulation response when target phonemes conflict with the source facial dynamics, together with generalization under pose variation, occlusion, rapid motion, and diverse real-world conditions.",
    },
  };
  const subgroupCopy = {
    multilingual: {
      titleCn: "",
      description: "Self-driven reconstruction is evaluated on Chinese, English, Korean, Russian, and Japanese videos.",
    },
    "aigc-video": {
      titleCn: "",
      description: "MiniMax H3 generates paired video and audio, and the generated video is redrawn using its accompanying generated speech. This evaluates a secondary visual-dubbing stage on AIGC content and pre-existing oral artifacts.",
      emptyText: "AIGC-generated video samples will be added here.",
    },
    "large-pose": {
      titleCn: "",
      description: "We test synchronization stability under profile views, head turns, and substantial pose changes.",
    },
    occlusion: {
      titleCn: "",
      description: "We examine reconstruction quality and temporal stability when the mouth or face is occluded by hands or other objects.",
    },
    "rapid-head-motion": {
      titleCn: "",
      description: "We assess robustness under rapid head turns, camera shake, and other challenging high-frequency motion.",
    },
    "in-the-wild": {
      titleCn: "",
      description: "We evaluate generalization across diverse identities, scenes, compositions, and capture conditions.",
    },
  };
  en.resultSettings.forEach((setting) => {
    Object.assign(setting, settingCopy[setting.id]);
    setting.subgroups.forEach((subgroup) => Object.assign(subgroup, subgroupCopy[subgroup.id]));
  });

  const caseDescriptions = {
    "self-01": "Chinese self-driven reconstruction. The input video and driving speech come from the same real video, enabling direct comparison with the ground truth.",
    "self-english-01": "English self-driven reconstruction. The input video and driving speech come from the same real video, enabling direct comparison with the ground truth.",
    "self-korean-01": "Korean self-driven reconstruction. The input video and driving speech come from the same real video, enabling direct comparison with the ground truth.",
    "self-russian-01": "Russian self-driven reconstruction. The input video and driving speech come from the same real video, enabling direct comparison with the ground truth.",
    "self-japanese-01": "Japanese self-driven reconstruction. The input video and driving speech come from the same real video, enabling direct comparison with the ground truth.",
    "self-aigc-01": "Self-driven reconstruction on AIGC-generated content, evaluating adaptation to synthetic portraits and generated visual details.",
    "self-aigc-02": "Self-driven reconstruction on an AIGC-generated livestream-room sequence.",
    "self-aigc-03": "Self-driven reconstruction on an AIGC-generated office sequence.",
    "self-aigc-04": "Self-driven reconstruction on an AIGC-generated private-study sequence.",
    "self-aigc-05": "Self-driven reconstruction on an AIGC-generated art-studio sequence.",
    "cross-large-pose": "Cross-audio driving under profile views, head turns, and substantial pose variation.",
    "cross-large-pose-02": "A second cross-audio case featuring profile views, head turns, and substantial pose variation.",
    "cross-occlusion-01": "Cross-audio lip editing under real-world occlusion, compared with four baseline methods.",
    "cross-occlusion-02": "A second real-world occlusion case highlighting mouth boundaries, identity preservation, and temporal stability.",
    "cross-occlusion-03": "A third real-world occlusion case comparing lip synchronization, identity preservation, and temporal stability.",
    "cross-rapid-head-motion-01": "Cross-audio driving under rapid head motion and high-frequency shake, testing articulation stability, identity preservation, and temporal consistency.",
    "cross-rapid-head-motion-02": "A second rapid-motion case for further evaluation of articulation stability, identity preservation, and temporal consistency under challenging dynamics.",
    "cross-in-the-wild-01": "Cross-audio driving on a randomly sampled real-world video with natural illumination, composition, and subject motion.",
    "cross-in-the-wild-02": "A second in-the-wild case comparing lip synchronization, identity preservation, and temporal stability.",
    "cross-in-the-wild-03": "A third in-the-wild case demonstrating generation quality and robustness in a complex real-world scene.",
    "cross-in-the-wild-04": "A fourth in-the-wild case demonstrating cross-audio driving across diverse real-world conditions.",
    "cross-in-the-wild-05": "A fifth in-the-wild case further demonstrating generation quality and robustness in challenging real-world content.",
  };
  en.qualitativeResults.forEach((item) => {
    item.description = caseDescriptions[item.id] || item.description;
  });

  en.method = {
    title: "Method",
    intro: "TBDub builds on the mask-free, reference-conditioned video editor of X-Dub/Wan. The complete source video and motion context constrain identity and spatiotemporal structure, while frame-aligned speech controls target articulation. Production-domain post-training yields a 30-step Teacher, which task-aware DMD2 compresses into a two-step Student.",
    pipelineAlt: "TBDub pipeline: a VAE encodes the source video into reference and target latents, motion context is injected into the target sequence, and a DiT jointly models reference tokens, noisy target tokens, and multi-layer HuBERT driving-audio features before VAE decoding.",
    facts: [
      { value: "X-Dub/Wan", label: "Inherited mask-free backbone" },
      { value: "78,161", label: "Currently recorded retained pseudo pairs" },
      { value: "30 → 2", label: "Teacher / student steps" },
      { value: "42.49×", label: "DiT-stage speedup (H20)" },
    ],
    pipeline: [
      {
        index: "01",
        title: "Source video",
        meta: "TRACK · CROP · 3D VAE",
        text: "Encode the complete reference video into spatiotemporal latents. Inherited motion context fixes part of the target latents to constrain identity, pose, and local dynamics.",
        tone: "visual",
      },
      {
        index: "02",
        title: "Driving audio",
        meta: "HUBERT LARGE · L9–12",
        text: "Fuse multi-layer HuBERT speech features and construct a local audio window for every video-latent frame.",
        tone: "audio",
      },
      {
        index: "03",
        title: "Joint DiT editor",
        meta: "REF + NOISY TARGET TOKENS",
        text: "The inherited X-Dub design jointly models reference and target tokens with separate timesteps and 3D RoPE. Self-attention integrates spatiotemporal context, while audio and text cross-attention operate on the target branch.",
        tone: "featured",
      },
      {
        index: "04",
        title: "Flow prediction",
        meta: "CONDITIONAL VELOCITY FIELD",
        text: "Flow Matching predicts a conditional velocity field from noisy target latents and progressively recovers the edited target representation.",
        tone: "flow",
      },
      {
        index: "05",
        title: "Video output",
        meta: "VAE DECODE · COMPOSITE",
        text: "Decode the edited target tokens and composite the face sequence back into the source video while preserving non-target regions.",
        tone: "output",
      },
    ],
    details: [
      {
        index: "A",
        eyebrow: "TASK-RELEVANT PSEUDO PAIRS",
        title: "Break the reference-copy shortcut",
        text: "Direct self-reconstruction encourages the model to copy articulation from the reference and underuse audio. We use a redubbed video as the conditioning reference, the clean original as the target, and its aligned original speech as the target condition. Identity and scene remain matched while mouth motion differs, making speech necessary for recovering the correct articulation.",
        meta: ["89,462 candidates", "78,161 retained", "87.37% retention", "Audiovisual calibration"],
      },
      {
        index: "B",
        eyebrow: "PRODUCTION DATA & CONDITIONING",
        title: "Adapt to production data and recover oral detail",
        text: "Post-training samples livestream and green-screen studio data at a 3:1 ratio. Lighting and oral degradation affect only the conditioning reference while the target stays clean; an identity-matched oral prior restores local texture, and multi-layer HuBERT features provide frame-aligned speech conditioning.",
        meta: ["Livestream : studio = 3 : 1", "Conditioning-only augmentation", "Clean oral prior", "HuBERT-large L9–12"],
      },
      {
        index: "C",
        eyebrow: "WEIGHTED FLOW MATCHING",
        title: "Focus Teacher learning on faces and temporal stability",
        text: "The 30-step Teacher retains flow matching while increasing training weights for face and mouth regions, later frames, and the cold-start first latent when motion context is absent. This concentrates supervision on oral detail and long-range temporal stability.",
        meta: ["Face / mouth weighting", "Later-frame weighting", "Cold-start first latent", "Motion dropout"],
      },
      {
        index: "D",
        eyebrow: "TASK-AWARE TWO-STEP DISTILLATION",
        title: "Compress guided generation into two single-route steps",
        text: "A guided Teacher score is constructed from X-Dub's unconditional, reference-only, and reference-plus-audio dynamic-CFG routes, while Student and FakeScore are trained with consistent conditions. The Student retains a differentiable two-step trajectory with one fully conditioned DiT forward per step; region-selective warmup and FP32 EMA stabilize few-step training.",
        formula: "30 steps × 3 guidance routes  →  2 steps × 1 conditional route",
        meta: ["Conditional DMD2", "2,000-step selective warmup", "FP32 EMA · β = 0.97", "5 FakeScore : 1 Student"],
      },
    ],
  };

  Object.assign(en.quantitativeResults.limitation, {
    title: "Full-reference metrics describe matched reconstruction, not the complete dubbing task",
    text: "PSNR, SSIM, LPIPS, FID, and CSIM require frame-aligned ground truth and primarily measure matched reconstruction, perceptual quality, distribution realism, and identity preservation in the self-driven setting. Cross-driven dubbing has no unique frame-aligned mouth target, so SyncNet metrics, Response Mouth PSNR, subjective MOS, and direct video comparison provide complementary evidence. Objective metrics do not perfectly correlate with human perception, and no single reconstruction or synchronization score can replace inspection of cross-audio articulation, naturalness, and robustness under challenging conditions.",
  });

  en.quantitativeResults.readingNotes = [
    {
      eyebrow: "01 · PAIRED QUALITY",
      title: "PSNR / SSIM / LPIPS",
      text: "These metrics are computed only on the self-driven TalkVid subset, where the source video provides frame-aligned ground truth. Higher PSNR and SSIM are better, whereas lower LPIPS is better. Face- and mouth-region LPIPS help separate overall facial fidelity from localized oral degradation.",
    },
    {
      eyebrow: "02 · NO-REFERENCE SYNC",
      title: "LSE-C / LSE-D",
      text: "SyncNet searches over temporal offsets to identify the best audio–visual alignment. LSE-C measures how distinctly the best offset stands out and is better when higher; LSE-D measures the embedding distance at that offset and is better when lower. The two metrics should be interpreted jointly.",
    },
    {
      eyebrow: "03 · DISTRIBUTION QUALITY",
      title: "FID",
      text: "FID measures the feature-distribution distance between real and generated frames and is better when lower. It reflects distribution-level visual realism rather than identity preservation or lip synchronization. With only 38 samples, it should be treated as supporting evidence.",
    },
    {
      eyebrow: "04 · IDENTITY PRESERVATION",
      title: "ArcFace CSIM",
      text: "We extract ArcFace embeddings (antelopev2 / glintr100) after five-point face alignment and compute cosine similarity between generated and ground-truth frames. Higher values indicate stronger identity preservation. Methods that make minimal visual changes naturally approach one, so CSIM should not be ranked in isolation.",
    },
  ];

  en.quantitativeResults.protocol = [
    {
      name: "PSNR",
      text: "Computed as 10·log10(255² / MSE) in dB. An increase of roughly 6 dB corresponds to halving the root-mean-square pixel error. PSNR tends to favor small edit regions and blurred outputs.",
    },
    {
      name: "SSIM",
      text: "Compares luminance, contrast, and structural relationships within local windows on the luminance channel. It ranges from 0 to 1 and still requires frame-aligned ground truth.",
    },
    {
      name: "LPIPS",
      text: "Measures perceptual distance between generated and ground-truth frames using deep features from a pretrained network. Lower is better; face and mouth scores are reported separately, with all 38 samples successfully evaluated.",
    },
    {
      name: "Regions",
      text: "The face region uses the median face box over the sequence. The mouth region spans 25%–75% horizontally and 55%–95% vertically within that box. Full-frame scores are easily dominated by unedited background pixels.",
    },
    {
      name: "SyncNet",
      text: "Mouth crops are resized to 224×224 and processed in five-frame windows. Visual and MFCC audio embeddings are compared over offsets of ±15 frames; LSE-C and LSE-D jointly characterize alignment.",
    },
    {
      name: "FID",
      text: "Compares Inception feature distributions between real and generated frames. Lower values indicate a closer global visual distribution; FID does not assess frame-wise reconstruction, identity preservation, or audio–visual synchronization.",
    },
    {
      name: "ArcFace CSIM",
      text: "Uses antelopev2 / glintr100 to extract embeddings from five-point-aligned faces and computes cosine similarity between generated and ground-truth frames. Higher values indicate closer identity features.",
    },
  ];

  Object.assign(en.subjectiveResults, {
    intro: "Three raters score six anonymized and independently randomized method outputs on the same 38 TalkVid clips using a 0–5 MOS scale. Each method receives 114 rating records. Ground truth is shown only to support judgments of identity and overall appearance and is not rated.",
    note: "Only mean MOS values are currently reported, without statistical significance tests; small differences such as 0.02 are therefore not interpreted as significant improvements.",
    analysisTitle: "Subjective gains are largely preserved after two-step distillation",
    analysis: "Relative to X-Dub, the TBDub Teacher improves lip-sync, identity, and visual-quality MOS by 0.14, 0.95, and 0.90, with the largest gains in identity consistency and overall visual quality. The two-step Student scores 3.85, 3.72, and 3.80 on the three criteria: it obtains the highest mean lip-sync and visual-quality scores, while its identity score is only 0.06 below the Teacher. These results indicate that subjective generation quality is largely retained despite the substantial reduction in sampling steps.",
  });

  Object.assign(en.efficiency, {
    scope: "Measurements use a single NVIDIA H20 GPU at 512×512. Timing starts at the first VAE encoding operation and ends after the final VAE decoding operation. Model loading, HuBERT audio encoding, face detection and cropping, color correction, compositing, video encoding, and file writing are excluded. The reported values therefore measure VAE-to-VAE core generation throughput rather than complete application-level end-to-end speed.",
  });
  en.efficiency.summary = [
    { name: "STUDENT THROUGHPUT", value: "7.13 FPS", note: "Effective FPS · 376 valid frames" },
    { name: "CORE PIPELINE SPEEDUP", value: "13.93×", note: "734.75 s → 52.76 s" },
    { name: "DIT SPEEDUP", value: "42.49×", note: "696.77 s → 16.40 s" },
  ];

  window.PROJECT_DATA_EN = en;
  window.UI_COPY = {
    zh: {
      languageLabel: "语言选择",
      backTopLabel: "回到顶部",
      pageNavLabel: "页面导航",
      pageDescription: "TBDub — 面向生产的视觉配音",
      watchDemos: "▶ 观看演示",
      overviewDescription: "生产场景中的关键失效模式，以及 TBDub 相对 X-Dub 的任务自适应改进。",
      promiseText: "以生产域后训练提升质量与鲁棒性，以任务感知两步蒸馏显著降低核心生成成本。",
      pipelineLabel: "端到端流程",
      pipelineFlow: "参考视频 + 驱动音频 → 配音视频",
      pipelineCaption: "该生成主干继承自 X-Dub/Wan：完整参考视频与 motion context 提供身份和时空约束，帧对齐音频控制目标分支中的语音相关动态。",
      trainingTitle: "生产域适配与两步蒸馏",
      trainingIntro: "TBDub 包含两个连续阶段：任务自适应后训练获得 30 步 Teacher，再以条件 DMD2 将其压缩为 2 步 Student。",
      evaluationIntro: "在 TalkVid 随机抽取的 38 条样本上，对 30 步 Teacher、2 步 EMA Student 与代表性方法进行客观指标和主观 MOS 比较。",
      efficiencyTitle: "推理效率",
      efficiencyIntro: "两步 Student 在显著降低 DiT 计算量的同时，基本保持 Teacher 的视觉质量、身份一致性与音画同步。",
      citationDescription: "TBDub 技术报告引用信息。",
      footerText: "研究项目页面 · 基于 GitHub Pages",
      backToTop: "返回顶部 ↑",
      testCategory: "测试类别",
      evaluationSetting: "评测设置",
      comingSoon: "即将补充",
      emptySamples: "样例将在此处补充。",
      pendingVideoLabel: "视频待添加",
      ready: "已就绪",
      pending: "待添加",
      reference: "参考",
      tableCaption: "38 个样本上的配对画质、分布质量、身份保持、音画同步与口部响应结果",
      tableFootnote: "<strong>指标范围。</strong> PSNR / SSIM 与 face / mouth LPIPS 衡量逐帧重建质量，FID 衡量整体分布真实感，CSIM 衡量身份保持，LSE-C / LSE-D 衡量音画同步；Response mouth PSNR 反映替换音频前后的口部响应幅度，不是越高越好的质量指标。GT 只作为 LSE 参考，不参与重建排名。",
      subjectiveEyebrow: "主观评测",
      subjectiveTitle: "MOS 用户研究",
      subjectiveCaption: "38 条 TalkVid 视频上的主观 MOS 结果",
      throughputTitle: "方法吞吐对比",
      throughputCaption: "单张 NVIDIA H20、512×512 下的核心生成吞吐",
      profileTitle: "Teacher–Student 分阶段耗时",
      profileCaption: "同一条 376 有效帧样本上的配对速度分析",
      timingScope: "计时范围",
      copy: "复制",
      copied: "已复制",
      selectText: "请手动选择",
    },
    en: {
      languageLabel: "Language selection",
      backTopLabel: "Back to top",
      pageNavLabel: "Page navigation",
      pageDescription: "TBDub — Production-Oriented Visual Dubbing",
      watchDemos: "▶ Watch demos",
      overviewDescription: "Production failure modes and the task-adaptive improvements introduced over X-Dub.",
      promiseText: "Improve quality and robustness through production-domain post-training, then reduce core generation cost through task-aware two-step distillation.",
      pipelineLabel: "END-TO-END PIPELINE",
      pipelineFlow: "REFERENCE VIDEO + DRIVING AUDIO → DUBBED VIDEO",
      pipelineCaption: "The generative backbone is inherited from X-Dub/Wan: the complete reference video and motion context constrain identity and spatiotemporal structure, while frame-aligned audio controls speech-related target dynamics.",
      trainingTitle: "Production adaptation and two-step distillation",
      trainingIntro: "TBDub uses two consecutive stages: task-adaptive post-training produces a 30-step Teacher, and conditional DMD2 compresses it into a two-step Student.",
      evaluationIntro: "We compare the 30-step Teacher, two-step EMA Student, and representative methods through objective metrics and subjective MOS on 38 randomly sampled TalkVid clips.",
      efficiencyTitle: "Efficiency",
      efficiencyIntro: "The two-step Student sharply reduces DiT computation while largely retaining the Teacher's visual quality, identity consistency, and audiovisual synchronization.",
      citationDescription: "Citation information for the TBDub technical report.",
      footerText: "Research project page · Built for GitHub Pages",
      backToTop: "Back to top ↑",
      testCategory: "TEST CATEGORY",
      evaluationSetting: "EVALUATION SETTING",
      comingSoon: "COMING SOON",
      emptySamples: "Samples will be added here.",
      pendingVideoLabel: "Video pending",
      ready: "READY",
      pending: "PENDING",
      reference: "Reference",
      tableCaption: "Paired visual quality, distribution quality, identity preservation, synchronization, and response results on 38 samples",
      tableFootnote: "<strong>Metric scope.</strong> PSNR / SSIM and face / mouth LPIPS measure frame-wise reconstruction quality; FID measures distribution realism; CSIM measures identity preservation; and LSE-C / LSE-D assess audio–visual synchronization. Response mouth PSNR is a diagnostic of editing magnitude after replacing the speech and is not a quality metric to maximize. GT is included only as an LSE reference and is excluded from reconstruction ranking.",
      subjectiveEyebrow: "SUBJECTIVE EVALUATION",
      subjectiveTitle: "MOS user study",
      subjectiveCaption: "Subjective MOS results on 38 TalkVid clips",
      throughputTitle: "Cross-method throughput",
      throughputCaption: "Core generation throughput on one NVIDIA H20 at 512×512",
      profileTitle: "Teacher–Student stage-wise profile",
      profileCaption: "Paired timing analysis on one 376-valid-frame sample",
      timingScope: "TIMING SCOPE",
      copy: "Copy",
      copied: "Copied",
      selectText: "Select text",
    },
  };
})();
