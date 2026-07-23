 Role: You are an expert AI/ML educator and technical communicator about training large AI models at scale and loading
  large datasets.

  Task: You are to create a set of progressive learning, step-by-step tutorials in the folder `tutorials about how to use
  the Stanford HAIC SLURM cluster as well as the GaitLU-1M dataset together. Start with no assumption about theuser who
  is trying to learn about GPU computing and large-scale data extraction.

  Systematically and progressively educate the user from the basics of setting up the HAIC environment to extracting the
  GaitLU-1M dataset, which is stored in `../../Downloads/gaitlu-1m`. Since the user is operating on a laptop, the current
  environment is not sufficient for storing the entire GaitLU-1M dataset. Instead, teach the user from first principles
  how to utilize the HAIC environment properly for training CoDy-JEPA. The user is currently on Week 2 of `tutorials/
  cody-jepa-10-week-plan.md`. Once you finish explaining how to store the data, carefully and thoughtfully walk the user
  to creating a loader returning [B, T, C, H, W] batches, deterministic validation sampling, metadata summaries, split
  manifests, batch visualization, frame-difference or motion-energy diagnostics, and dummy probe exports. Use clear
  examples and specific code snippets to clearly illustrate the various concepts and ideas related to building a
  trustworthy data pipeline.

  Your writing must be clear, fluent, connecting ideas, and without using em-dashes. Each progression should be
  accompanied by a Jupyter notebook that references realistic datasets. Review each cell of the notebook to further
  clarify and simplify the explanation of code in each cell -- possibly by splitting further to create more atomic unit
  of processing and explanations.

  Your gate for continuing or pivoting is to continue when batches and splits are reproducible. If the full dataset is
  fragile, reduce to a controlled subset and preserve split discipline. Use the following starting sources to ground your
  reasoning:

  - SLURM: https://stanford-rc.github.io/docs-earth/docs/slurm-basics#pancakes-code
  - GaitLU-1M: https://github.com/ShiqiYu/OpenGait/blob/master/datasets/GaitLU-1M/README.md

  Use codex:adversarial-review to review your implementation, and fix all suggested actions and corrections.

  For managing libraries and dependencies, we should use uv exclusively: update all docs and notebooks.

  Use fan out subagents and dynamic workflows to orchestrate your tasks.