# 原库来源与版本约定

原库使用已有的压缩 JSONL 正文分片。一行至少提供 `paper_id`、`text`、`text_sha256`。只读 SQLite 表 `paper` 提供：

```sql
paper(paper_id TEXT PRIMARY KEY, shard TEXT, offset INTEGER, length INTEGER, row INTEGER)
```

`offset` 和 `length` 是该 JSON 行在解压数据流中的字节位置。`row` 为零起点记录号。已有 gzip 寻址索引名为 `<shard>.gzidx`；读取器不会创建它。

本地配置示例：

```json
{
  "corpus": {
    "body_root": "/path/to/existing/body-shards",
    "locator_db": "/path/to/existing/locators.sqlite",
    "index_root": "/path/to/existing/gzip-indexes"
  },
  "catalog": "/path/to/source_catalog.local.json",
  "base_url": "http://127.0.0.1:8769"
}
```

配置可通过 `--source-config` 或环境变量 `THEOREM_SOURCE_CONFIG` 指定。空目录的初始来源目录表可写成 `{"sources":{},"version_bindings":{},"legacy_paths":{}}`。

来源读取顺序：

1. 按论文编号读原库现有记录并校验全文哈希。
2. 输入未指定具体版本时，以“版本未知的原库快照”展示，不擅自写成最新版。
3. 输入指定版本时，要求原生下载记录绑定的版本及完整原始源码比较。只规范换行；数据集附加书目时，额外部分须与原生源码包中校验过的完整书目成员吻合。
4. 无法绑定所需版本时，读取既有历史版本缓存。缺失就返回 `missing`，没有隐式下载或全库扫描。

完整源码一致说明原库可提供所请求版本的 TeX 内容；若几个版本源码完全相同，不能据此唯一识别原库归档的版次。算法中的时间顺序仍来自输入原生版本资料，不由正文相等重新推断。

`source_access.backend` 区分 `corpus`、`historical_cache`、`legacy_archive`、`missing`。`corpus_locator` 和 SHA-256 可以复核原库位置；`viewer_url` 是本地阅读入口。原库压缩文件路径不是可以套上 TeX 行号的文本文件。

```bash
python -X utf8 tex_sources.py --config /path/to/config.json read --ref corpus-REF --start-line 100 --end-line 140
```

行号由实际读取的全文计算。精确声明匹配、标签包围环境及注释/空白规范后的唯一匹配分别标记；它们只用于导航，不证明命题对应或版本身份。

清理旧展开副本时，旧算法曾扫描的精确字节集中压缩保留。已有算法结果不重写，阅读视图覆盖新的导航指针；原来的扫描哈希不改成原库哈希。冻结的旧报告也保存原始字节，以便核对旧读者记录。

默认全文上限为 8 MiB，单个压缩正文记录上限为 32 MiB，正文缓存总计最多 16 MiB。HTTP 阅读器只绑定 `127.0.0.1`，只接受目录表中的来源编号，不能通过网址任意读取文件。
