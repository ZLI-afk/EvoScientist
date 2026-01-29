<!-- Add logo here -->
<h1 align="center">
  <!-- <img src="./assets/CCD_icon_logo.png" alt="CCD Logo" height="27" style="position: relative; top: -2px;"/> -->
  <strong>EvoScientist</strong>
</h1>


<div align="center">

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&width=435&lines=Towards+Self-Evolving+AI+Scientists+for+End-to-End+Scientific+Discovery" alt="Typing SVG" /></a>

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logo=googlelens&logoColor=4285F4)]()
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxx-b31b1b?style=for-the-badge&logo=arxiv&logoColor=b31b1b)]()
[![Gradio Demo](https://img.shields.io/badge/Gradio-Online_Demo-FFCC00?style=for-the-badge&logo=gradio&logoColor=yellow&labelColor=grey)]()
[![Evaluation Split](https://img.shields.io/badge/HF-Test_Dataset-AECBFA?style=for-the-badge&logo=huggingface&logoColor=FFCC00&labelColor=grey)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

</div>

## 🔥 News
> TODO
- **[27 Sep 2025]** ⛳ Our preprint is now live on [arXiv] — check it out for details.


## Overview
> TODO


## 📖 Contents
- [⛏️ Installation](#️-installation)
- [⚡ Quick Start](#-quick-start)
  - [CLI Inference](#cli-inference)
  - [Script Inference](#script-inference)
  - [Web Interface](#web-interface)
- [📊 Evaluation](#-evaluation)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📦 Codebase Contributors](#-codebase-contributors)
- [📜 License](#-license)

## ⛏️ Installation

> [!TIP]  
> Use [`uv`](https://pypi.org/project/uv) for installation — it's faster and more reliable than `pip`.
### For Development

```Shell
# Create and activate a conda environment
conda create -n EvoSci python=3.11 -y
conda activate EvoSci

# Install in development (editable) mode
pip install EvoScientist
# or
pip install -e .
```

### Option 1:
Install the latest version directly from GitHub for quick setup:
> TODO
### Option 2: 
If you plan to modify the code or contribute to the project, you can clone the repository and install it in editable mode:

> TODO

<details>
<summary> 🔄 Upgrade to the latest code base </summary>

```Shell
git pull
uv pip install -e .
```

</details>

## ⚡ Quick Start

### CLI Inference  
You can perform inference directly from the command line using our CLI tool:

![demo](./assets/EvoScientist_cli.png)

```Shell
python -m EvoScientist
```
**Optional arguments:**  

> TODO

### Script Inference
```python
from EvoScientist import EvoScientist_agent
from langchain_core.messages import HumanMessage
from EvoScientist.utils import format_messages

thread = {"configurable": {"thread_id": "1"}}
question = "Hi?"
last_len = 0

for state in EvoScientist_agent.stream(
    {"messages": [HumanMessage(content=question)]},
    config=thread,
    stream_mode="values",
):
    msgs = state["messages"]
    if len(msgs) > last_len:
        format_messages(msgs[last_len:]) 
        last_len = len(msgs)
```

<details>
<summary> Output </summary>

```json

╭─────────────────────────────────────────────────── 🧑 Human ────────────────────────────────────────────────────╮
│ Hi?                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────── 📝 AI ─────────────────────────────────────────────────────╮
│ Hi! I'm here to help you with experimental research tasks. I can assist with:                                   │
│                                                                                                                 │
│ - **Planning experiments** - designing stages, success criteria, and workflows                                  │
│ - **Running experiments** - implementing baselines, training models, analyzing results                          │
│ - **Research** - finding papers, methods, datasets, and baselines                                               │
│ - **Analysis** - computing metrics, creating visualizations, interpreting results                               │
│ - **Writing** - drafting experimental reports and documentation                                                 │
│                                                                                                                 │
│ What would you like to work on today?                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

### Web Interface  

> TODO


## 📊 Evaluation  

> TODO

## 📝 Citation

If you find our paper and code useful in your research and applications, please cite using this BibTeX:

> TODO

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**Deep Agents**](https://github.com/langchain-ai/deepagents) — A framework for building AI agents that can interact with various tools and environments.
- [**Deep Agents UI**](https://github.com/langchain-ai/deep-agents-ui) — A user interface for visualising and managing Deep Agents.

We thank the authors for their valuable contributions to the open-source community.


## 📦 Codebase Contributors

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://youganglyu.github.io/">
          <img src="https://youganglyu.github.io/images/profile.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yougang Lyu"/>
          <br />
          <sub><b>Yougang Lyu</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://x-izhang.github.io/">
          <img src="https://x-izhang.github.io/author/xi-zhang/avatar_hu13660783057866068725.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xi Zhang"/>
          <br />
          <sub><b>Xi Zhang</b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

For any enquiries or collaboration opportunities, please contact: [**youganglyu@gmail.com**](mailto:youganglyu@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.