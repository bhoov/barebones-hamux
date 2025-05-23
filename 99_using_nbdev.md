

# Reflections on `nbdev` as documentation

> Trial run using `nbdev` for [documentation
> only](https://nbdev.fast.ai/tutorials/docs_only.html)

My approach was to create a `docs_src/` folder in my project root,
import the code that I write using standard tools/test suites, and use
nbdev just to generate the docs and examples that I want to deploy
alongside my app.

I’m writing docs in `.qmd` format to avoid the headache of
jupyter+git+vim+AI assist in VSCode.

## Pros/cons

Following are the pros/cons of switching to using `nbdev` solely for the
documentation instead of also for the code

**Pros:**

- I avoid a necessary “transpilation” step converting notebooks to
  python files every time I make a change to my code
- I can use VSCode’s excellent [interactive
  python](https://code.visualstudio.com/docs/python/jupyter-support-py)
  mode to develop my codebase
- I get to keep my
- Can use all nbdev goodies to make my executable documentation/test mix
- Can keep my code as a simple python file in the original repo

**Cons:**

- README of original repo is different than the `nbs/index.qmd` (nbd)
- My docs/tests/code become scattered across my repo, some living in the
  documentation, others living in dedicated pytest files
- I don’t get the auto-generated `__all__` export
