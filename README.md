# 定理强度变化线索筛查

输入已经取得的两段数学声明及作者、版本、来源材料，程序用确定性规则指出新增条件、适用范围变化和修订说明，输出可复核的阅读线索。

重点是同作者后续版本中未被所查说明材料解释的条件收紧。人和模型是结果的读者；运行过程中没有模型调用、逐条人工裁决或人工真假标签。

当前数学核心为 **0.6.1**，来源读取器为 **1.0.0**，输出层为 **1.6.1**。本库供代码审阅和有限规则实验，不能用它认证论文错误、作者隐瞒或全库真实召回率。

## 直接运行

Python 3.11 以上；核心和合成测试只用标准库。

```bash
python -X utf8 -m unittest discover -v
python -X utf8 run_algorithm.py --input examples/pairs.jsonl --output-dir demo-output
```

打开 `demo-output/用户结果.md`，或读取 `results.jsonl` 与 `summary.jsonl`。示例是人工构造的数学文本，不对应真实论文。

示例来源路径相对于仓库根目录。要生成可点击的实际全文行号，可先运行 `python -X utf8 prepare_recheck99.py --input examples/pairs.jsonl --output-dir demo-evidence`，再把 `demo-evidence/inputs/provided.jsonl` 交给运行器。

## 程序做什么

- 保留作者姓名关联与实际身份消歧的区别，区分首发时间和具体版本时间。
- 先检查声明对应关系和竞争候选，再判断是否进入变化线索队列。
- 过滤有限规则能够完整解释的换序、区间换写、符号与文字换写，以及来源支持的字面宏别名。
- 区分公式前提、量词范围、结论、分段条件和枚举分项；只在有限的精确区间、数值和标准数集规则内比较局部范围。
- 对新增条件给出所在分项和相关保证公式，避免把局部估计条件当作整条定理的前提。
- 新增修饰词与假设从句按大小写敏感的同一对象比较；有限的属性搬移不算新增，无法处理的枚举或重复绑定留在未分类变化中。
- 渐近界仅在两端明确同一极限（正无穷或零）、相同记号、简单同一主语与结论角色时比较；零附近不比较对数因子。未知极限、嵌套条件和复杂表达式保留原始差异，不猜测强弱。
- 分别记录版本说明、公开纠错材料及其命题关联；论文级有勘误不自动解释当前命题。

阅读报告的首屏保留作者、实际版本日期、改前／改后片段及全文入口；总览先列具体变化，详细对齐依据和后文观察折叠展示。0.6.1 合入经反例修正的外部代码建议；旧版真实候选的冻结判定没有随此次代码更新重算，不能把旧评测数字当作新版准确率。

核心入口是 `strength_signals.analyze_pairs(pairs)`。输入是候选对，**候选检索的 BM25 索引不包含在这个代码库中**。已有缓存中的完整候选竞争信息可以随输入传入；不会因只选几条重放就假定不存在竞争候选。

## 原库优先的 TeX 来源

`tex_sources.py` 通过已有 SQLite 定位表和 gzip 寻址索引，按需读取原库正文，不重建索引、不把整库放入内存，也不生成新的正文 TeX 文件。

原库每篇保存一个快照，不能假定含有所有历史版本。已知版本须由原生版本记录及完整原始源码比较来绑定；某条定理文字相同不构成版本身份证明。未能绑定的历史版本沿用既有版本缓存。缺失时明确返回 `missing`，不会自行下载。

要使用本地全库读取器：

```bash
python -m pip install -r requirements-corpus.txt
python -X utf8 tex_sources.py --config /path/to/source_config.local.json serve
```

配置格式见 [来源约定](docs/SOURCES.md)。启动后，报告的本地网址显示完整 TeX 和实际行号；模型可使用 `tex_sources.py ... read` 按行读取。普通文件路径仍受支持。

增强任意已有候选的来源信息：

```bash
python -X utf8 prepare_recheck99.py --input /path/to/pairs.jsonl --output-dir evidence --source-config /path/to/source_config.local.json
python -X utf8 run_algorithm.py --input evidence/inputs/provided.jsonl --output-dir checked
```

`prepare_recheck99.py` 的文件名来自早期实验；公开入口使用明确的 `--input` 和输出目录。官方说明资料默认只读本地缓存；`--fetch-metadata` 是另行显式请求的摘要页获取，不是 TeX 下载。

## 请从哪里审代码

[REVIEW_GUIDE.md](REVIEW_GUIDE.md) 给出模块关系、评审重点和已知缺口。[评测边界](docs/EVALUATION.md) 区分规则测试、候选复验和真实漏检率。来源清理的操作脚本在 `tools/migrate_tex_sources.py`，它针对原实验目录布局，公开在这里用于审阅迁移与恢复逻辑；运行它需要自有实验材料和显式本地配置。

本库提供代码、合成测试和合成样例。论文全库、实际论文全文、个人本地配置和逐案人工标签由使用者自行管理。
