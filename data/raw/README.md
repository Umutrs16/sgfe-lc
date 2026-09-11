# 902 野外样点（核心真值）

请将官方元数据文件放入本目录，**不必下载 11.9 GB 照片**：

- `CA_LC_Sites&PhotosInfo.rar`（约 321 KB）
- 或解压后的 `.xlsx` / `.shp` / `.csv` / `.gpkg`

获取地址：

- 干旱区生态与资源科学数据中心：https://data.dcxjegi.cn/portal/metadata/12bc429a-085d-47fa-bbab-e12081934b92
- 全球变化数据仓储：DOI [10.3974/geodb.2026.06.06.V1](https://doi.org/10.3974/geodb.2026.06.06.V1)

放入后运行：

```bash
python scripts/00_audit_sites.py
```

脚本会自动识别经纬度、年份、国家、地类字段，并写出标准表 `data/processed/sites_harmonized.gpkg`。
