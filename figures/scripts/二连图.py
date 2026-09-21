import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter, FormatStrFormatter

import pandas as pd

import math

import numpy as np

import re

import os

import glob



# ==========================================

# 第一部分：视觉参数配置 (保持修改版：加大加粗)

# ==========================================



# 1. 轴标题 (Recall, QPS) 的大小

AXIS_LABEL_FONTSIZE = 70

AXIS_LABEL_WEIGHT   = "bold"



# 2. 刻度数字 (0.8, 100, 1000) 的大小

TICK_LABEL_FONTSIZE = 70

TICK_LABEL_WEIGHT   = "bold"



# 3. 图例文字的大小

LEGEND_FONTSIZE     = 65



# 4. 子图标题 (a) Algorithms ... 的大小

SUBTITLE_FONTSIZE   = 70

SUBTITLE_WEIGHT     = "bold"
# 5. 线条和边框粗细
# 含义：每隔 5 个点画一个标记 (数字越大越稀疏)
MARKER_STEP = 3
LINE_WIDTH          = 6.0   # <--- 【修改】加大线宽 (原 3.5 -> 6.0)
MARKER_SIZE         = 25.0  # <--- 【新增】统一控制点的大小 (建议设为线宽的 2~3 倍)
SPINE_WIDTH         = 4.0

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.serif"] = ["Times New Roman"]



COLORS = {

    "milvus-gpu-ivfflat": "#1f77b4",

    "milvus-hnsw": "#ff7f0e",

    "qdrant": "#9467bd",

    "weaviate-hnsw": "#d62728",

    "iRangeGraph": "#d62728",

    "acorn": "#17becf",              # 青色 (Cyan)

    "rangepq": "#e377c2",

    "Faiss-HNSW": "#1f77b4",  

    "HNSWLib": "#ff7f0e",      

    "Faiss-IVFPQ": "#2ca02c",

    "results": "#9467bd",

    "redis-hnsw": "#8c564b",

}



MARKERS = {

    "milvus-gpu-ivfflat": "v",

    "milvus-hnsw": "P",

    "qdrant": "o",

    "weaviate-hnsw": "s",

    "redis-hnsw": "*",

    "iRangeGraph": "o",

    "Faiss-HNSW": "s",  

    "HNSWLib": "^",      

    "Faiss-IVFPQ": "D",

    "results": "X",

}

def smart_y_format(x, pos):
    """
    如果是整数，就转成 int 显示 (100.0 -> 100)
    如果是小数，就保留原样显示 (0.1 -> 0.1)
    """
    if x.is_integer():
        return f"{int(x)}"
    else:
        return f"{x}"

# ==========================================

# 第二部分：核心逻辑

# ==========================================



def create_pareto_frontier(points):

    points.sort(key=lambda x: x[0], reverse=True)

    xs, ys = [], []

    max_qps_so_far = -1

    for r, q in points:

        if q > max_qps_so_far:

            xs.append(r)

            ys.append(q)

            max_qps_so_far = q

    frontier = sorted(zip(xs, ys), key=lambda x: x[0])

    if not frontier: return [], []

    return zip(*frontier)



# ==========================================

# 第三部分：数据解析 (修复版：已添加 ACORN 解析逻辑)

# ==========================================

def parse_all_data():

    all_data = []

    # 获取脚本所在的绝对路径

    script_dir = os.path.dirname(os.path.abspath(__file__))

   

    print(f"正在扫描目录: {script_dir} ...")

   

    # -------------------------------------------------

    # 1. 自动处理所有 .csv, .pkl, .xlsx 文件

    # -------------------------------------------------

    data_files = glob.glob(os.path.join(script_dir, "*.csv")) + \
        glob.glob(os.path.join(script_dir, "*.pkl")) + \
            glob.glob(os.path.join(script_dir, "*.xlsx"))



    if not data_files:

        print("【警告】在该目录下没有找到任何数据文件 (.csv/.pkl/.xlsx)！")



    for fpath in data_files:

        fname = os.path.basename(fpath)

        try:

            # --- 1. CSV 文件处理 ---

            if fname.endswith('.csv'):

                fname_lower = fname.lower()

                is_artificial = "artificial" in fname_lower or "artficial" in fname_lower

               

                if is_artificial:

                    df = pd.read_csv(fpath, header=None)

                    if df.shape[1] >= 4:

                        # 尝试根据列数智能推断 (通常 2 是 recall, 3 是 QPS)

                        if 3 in df.columns:

                            df.rename(columns={1: 'recall', 2: 'qps'}, inplace=True) # 假设 1=Recall, 2=QPS

                            if 'qps' not in df.columns and 3 in df.columns: # 如果上面不对，尝试 3=QPS

                                df.rename(columns={3: 'qps'}, inplace=True)

                        else:

                             df.rename(columns={1: 'recall', 2: 'qps'}, inplace=True)



                        if 'dataset' not in df.columns:

                            df['dataset'] = os.path.splitext(fname)[0]

                       

                        if "mixed" in fname_lower:

                            df['algorithm'] = 'iRangeGraph'

                        elif 'algorithm' not in df.columns:

                            df['algorithm'] = os.path.splitext(fname)[0]

                    else:

                        df = pd.read_csv(fpath)

                else:

                    df = pd.read_csv(fpath)



            # --- 2. Excel 文件处理 ---

            elif fname.endswith('.xlsx'):

                df = pd.read_excel(fpath)

                print(f"[成功] 读取Excel: {fname}")



            # --- 3. Pickle 文件处理 ---

            else:

                raw_data = pd.read_pickle(fpath)

                if isinstance(raw_data, dict) and not 'dataset' in raw_data:

                    temp_records = []

                    base_name = os.path.splitext(fname)[0].lower()

                    if base_name.startswith("acorn"): algo_val = "acorn"

                    elif base_name.startswith("digra"): algo_val = "digra"

                    elif base_name.startswith("rangepq"): algo_val = "rangepq"

                    else: algo_val = base_name

                   

                    for ds_key, points in raw_data.items():

                        if isinstance(points, list):

                            for pt in points:

                                if isinstance(pt, (list, tuple)) and len(pt) >= 2:

                                    temp_records.append({

                                        "algorithm": algo_val,

                                        "dataset": ds_key,

                                        "recall": pt[0],

                                        "qps": pt[1]

                                    })

                    df = pd.DataFrame(temp_records)

                elif isinstance(raw_data, list):

                    df = pd.DataFrame(raw_data)

                else:

                    df = raw_data

           

            # --- 通用数据清洗 ---

            df.columns = [str(c).strip().lower() for c in df.columns]

           

            rename_map = {

                'k-nn': 'recall',

                'knn': 'recall',

                'recall@1': 'recall'

            }

            df.rename(columns=rename_map, inplace=True)

           

            if {'dataset', 'recall', 'qps'}.issubset(df.columns):

                if 'algorithm' not in df.columns:

                    algo_name = os.path.splitext(fname)[0]

                    if "iRangeGraph" in fname: algo_name = "iRangeGraph"

                    elif fname == "results.pkl": algo_name = "results"

                    algo_name = algo_name.replace("-multi_vector", "").replace(".csv", "")

                    df['algorithm'] = algo_name

               

                df['recall'] = pd.to_numeric(df['recall'], errors='coerce')

                df['qps'] = pd.to_numeric(df['qps'], errors='coerce')

               

                records = df[['algorithm', 'dataset', 'recall', 'qps']].dropna().to_dict('records')

                all_data.extend(records)

                print(f"[成功] 读取文件: {fname} (包含 {len(records)} 条数据)")

            else:

                missing = {'dataset', 'recall', 'qps'} - set(df.columns)

                # print(f"[跳过] 文件 {fname} 缺少列: {missing}")

               

        except Exception as e:

            # print(f"[错误] 无法读取文件 {fname}: {e}")

            pass



    # -------------------------------------------------

    # 2. 解析 Log 文件 (这里是重点修改区域！！！)

    # -------------------------------------------------

    log_files = {

        'Faiss-HNSW': ['faiss_hnsw.log', 'faiss_hnsw_151'],

        'Faiss-IVFPQ': ['faiss_ivfpq_dynamic.log', 'faiss_ivfpq_151'],

        'HNSWLib': ['hnswlib.log', 'bvbj_hnswlib_151'],

        'hnsw+post': ['hnsw_post.log', 'faiss_hnsw_151'],

        # 【修改点】确保 ACORN 被包含在内

        'ACORN': ['acorn.log', '*gamma*', '*acorn*', '*_efc*', '*_efs*', 'output_*']

    }

   

    # 防止重复读取

    processed_files = set()



    for algo_name, patterns in log_files.items():

        matched_files = []

        for pattern in patterns:

            found = glob.glob(os.path.join(script_dir, pattern + "*"))

            if not found and os.path.exists(os.path.join(script_dir, pattern)):

                found = [os.path.join(script_dir, pattern)]

            matched_files.extend(found)

       

        matched_files = list(set(matched_files))



        for fpath in matched_files:

            if fpath in processed_files: continue

           

            fname = os.path.basename(fpath)

            try:

                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:

                    content = f.read()

               

                lines = content.split('\n')

                count = 0

               

                # ========================================================

                # 🔥🔥🔥 分支 A: 标准格式 (包含 [RESULT]) 🔥🔥🔥

                # ========================================================

                if '[RESULT]' in content:

                    for line in lines:

                        if '[RESULT]' in line:

                            ds = re.search(r'dataset=([a-zA-Z0-9_]+)', line)

                            rec = re.search(r'(?:recall|avg_recall@\d+)=([\d\.]+)', line)

                            qps = re.search(r'qps=([\d\.]+)', line)

                            if ds and rec and qps:

                                all_data.append({

                                    'algorithm': algo_name,

                                    'dataset': ds.group(1),

                                    'recall': float(rec.group(1)),

                                    'qps': float(qps.group(1))

                                })

                                count += 1

               

                # ========================================================

                # 🔥🔥🔥 分支 B: ACORN 专用格式 (没有 [RESULT]) 🔥🔥🔥

                # ========================================================

                else:

                    # 1. 尝试从文件内容找 dataset: xxx

                    current_dataset = None

                    ds_match = re.search(r'dataset\s*[:=]\s*([a-zA-Z0-9_\-]+)', content, re.IGNORECASE)

                    if ds_match:

                        current_dataset = ds_match.group(1)

                    else:

                        # 核心补救：如果文件内容里没写 dataset，直接用【文件名】当前缀

                        current_dataset = os.path.splitext(fname)[0]



                    if current_dataset:

                        for line in lines:

                            # 匹配 QPS=6.4, recall@100=0.9874 这种格式

                            qps_match = re.search(r'QPS\s*=\s*([\d\.]+)', line)

                            rec_match = re.search(r'(?:recall|avg_recall)@\d+\s*=\s*([\d\.]+)', line)

                           

                            if qps_match and rec_match:

                                all_data.append({

                                    'algorithm': algo_name, # 强制标记为配置中的名字(如 ACORN)

                                    'dataset': current_dataset,

                                    'recall': float(rec_match.group(1)),

                                    'qps': float(qps_match.group(1))

                                })

                                count += 1



                if count > 0:

                    processed_files.add(fpath)

                    print(f"[成功] 读取Log: {fname} (包含 {count} 条数据)")

                   

            except Exception as e:

                pass



    # ==========================================

    # 数据集名称标准化

    # ==========================================

    df_temp = pd.DataFrame(all_data)



    def normalize_dataset_name(name):

        name = str(name).lower()

        if 'artficial' in name or 'artificial' in name:

            if '80' in name or '0.8' in name: return 'artificial-0.8'

            if '50' in name or '0.5' in name: return 'artificial-0.5'

            if '12' in name or '0.12' in name: return 'artificial-0.12'

            if '06' in name or '0.06' in name: return 'artificial-0.06'

            return name # 默认保留

           

        if 'ag_news' in name: return 'ag_news'

        if 'app_reviews' in name: return 'app_reviews'

        if 'amazon' in name: return 'amazon'

        if 'cc_news' in name: return 'cc_news'

        if 'msong' in name: return 'msong'

       

        if 'deep' in name: return 'deep1m'

        if 'sift' in name: return 'sift10m'

        if 'tiny' in name: return 'tiny5m'

        if 'dbpedia' in name: return 'dbpedia-entities'

       

        if 'img' in name: return 'img'

        if 'speech' in name: return 'speech'

        if 'gpt4web' in name: return 'gpt4web'

       

        return name



    if not df_temp.empty:

        df_temp['dataset'] = df_temp['dataset'].apply(normalize_dataset_name)

        df_temp['algorithm'] = df_temp['algorithm'].replace('hnsw+post', 'HNSWLib')



    return df_temp



# ==========================================

# 第四部分：绘图主函数 (修改版：支持自动颜色)

# ==========================================

def draw_plot(df, dataset_name, out_png_name):

    # 算法白名单

    TARGET_ALGORITHMS = [

        "RangePQ", "DiGRA", "ACORN", "iRangeGraph",

        "hnsw+post", "hnsw", "ivfpq",

        "milvus-hnsw", "milvus-ivfpq", "milvus-gpu-ivfpq", "milvus-gpu-cagra",

        "weaviate-hnsw", "qdrant",

        "vearch-ivfpq", "vearch-hnsw",

        "redis-hnsw", "elasticsearch-hnsw", "pgvector-hnsw",

        "digra_w1-4_s1-6",

        "FAISS-hnsw", "FAISS-ivfpq", "HNSWlib", "milvus-gpu-ivfflat"

    ]

    df_temp = df[df["dataset"] == dataset_name]

   

    # 🔥🔥🔥【诊断代码】🔥🔥🔥

    if dataset_name == "artificial-0.8":  

        raw_algos = df_temp["algorithm"].unique()

        print(f"\n[诊断报告] artificial-0.8 原始读取到的算法 ({len(raw_algos)}个):")

        print(raw_algos)

        ignored = [algo for algo in raw_algos if str(algo).lower() not in [t.lower() for t in TARGET_ALGORITHMS]]

        if ignored:

            print(f"⚠️ [警告] 以下算法被白名单过滤掉了: {ignored}")

    # 🔥🔥🔥【诊断结束】🔥🔥🔥



    df = df[df["algorithm"].astype(str).str.lower().isin([x.lower() for x in TARGET_ALGORITHMS])]

    df_subset = df[df["dataset"] == dataset_name]

    if df_subset.empty: return



    algorithms = df_subset["algorithm"].unique()

    print(f"正在绘制 {dataset_name} (包含: {algorithms})")

   

    sorted_algos = sorted(algorithms)
    CAT_DB = ["milvus", "weaviate", "qdrant", "vearch", "redis", "elastic", "pgvector"]

    CAT_LIB = ["faiss", "hnswlib"]

    def get_category(algo_name):

        name_lower = str(algo_name).lower()

        for db in CAT_DB:

            if db in name_lower: return "Databases"

        for lib in CAT_LIB:

            if lib in name_lower: return "Libraries"

        return "Algorithms"



    SPECIAL_DATASETS = ["ag_news", "app_reviews", "amazon", "cc_news"]

    is_special_task = dataset_name in SPECIAL_DATASETS



    FALLBACK_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    FALLBACK_MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']



    # === 核心绘图逻辑 ===

    if is_special_task:

        # 【情景 A】特殊数据集：画 1 行 3 列

        fig, axes = plt.subplots(1, 3, figsize=(30, 12))

        all_handles = []

        all_labels = []

        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.60, wspace=0.25)
       

        plot_configs = [

            {"title": "(a) Algorithms", "cat": "Algorithms"},

            {"title": "(b) Libraries",  "cat": "Libraries"},

            {"title": "(c) Databases",  "cat": "Databases"}

        ]



        for idx, config in enumerate(plot_configs):

            ax = axes[idx]

            target_cat = config["cat"]

            current_algos = [a for a in sorted_algos if get_category(a) == target_cat]

           

            for i, algo in enumerate(current_algos):

                algo_data = df_subset[df_subset["algorithm"] == algo]

                points = list(zip(algo_data["recall"], algo_data["qps"]))

                px, py = create_pareto_frontier(points)

                if not px: continue

                px, py = list(px), list(py)

               

                if algo in COLORS: use_color = COLORS[algo]

                else: use_color = FALLBACK_COLORS[sorted_algos.index(algo) % len(FALLBACK_COLORS)]

               

                if algo in MARKERS: use_marker = MARKERS[algo]

                else: use_marker = FALLBACK_MARKERS[sorted_algos.index(algo) % len(FALLBACK_MARKERS)]



                ax.plot(px, py, label=algo, color=use_color, marker=MARKER_SIZE,

                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9,markevery=MARKER_STEP)



            ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

            ax.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

            ax.set_yscale('log')

            y_formatter = ScalarFormatter(); y_formatter.set_scientific(False)

            ax.yaxis.set_major_formatter(FuncFormatter(smart_y_format))

            ax.set_title(config["title"], y=-0.25, fontsize=20, fontweight='bold')

            for spine in ax.spines.values():
                spine.set_linewidth(SPINE_WIDTH)

            handles, labels = ax.get_legend_handles_labels()

            all_handles.extend(handles)

            all_labels.extend(labels)



        if all_handles:

            by_label = dict(zip(all_labels, all_handles))

            fig.legend(handles=by_label.values(), labels=by_label.keys(),

                       loc="lower center", bbox_to_anchor=(0.50, 0.90),

                       ncol=4, frameon=False, fontsize=16, prop={'weight': 'bold', 'size': LEGEND_FONTSIZE})

    else:

        # 【情景 B】普通数据集

        fig, ax = plt.subplots(figsize=(6.6, 5.9))

        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.75)

       

        for i, algo in enumerate(sorted_algos):

            algo_data = df_subset[df_subset["algorithm"] == algo]

            points = list(zip(algo_data["recall"], algo_data["qps"]))

            px, py = create_pareto_frontier(points)

            if not px: continue

            px, py = list(px), list(py)



            if algo in COLORS: use_color = COLORS[algo]

            else: use_color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]

           

            if algo in MARKERS: use_marker = MARKERS[algo]

            else: use_marker = FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]



            ax.plot(px, py, label=algo, color=use_color, marker=use_marker,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9,markevery=MARKER_STEP)  # <--- 改这里



        ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

        ax.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

        ax.set_yscale('log')

        y_formatter = ScalarFormatter(); y_formatter.set_scientific(False)

        ax.yaxis.set_major_formatter(FuncFormatter(smart_y_format))

        for spine in ax.spines.values(): spine.set_linewidth(SPINE_WIDTH)

        for label in ax.get_xticklabels() + ax.get_yticklabels():

            label.set_fontweight('bold')



        handles, labels = ax.get_legend_handles_labels()

        if handles:

            n_items = len(handles)

            final_ncol = min(n_items, 3) if n_items > 0 else 1

            ax.legend(handles=handles, labels=labels, loc="lower center",

                      bbox_to_anchor=(0.50, 1.00), borderaxespad=0.4,

                      ncol=final_ncol, frameon=False, fontsize=LEGEND_FONTSIZE, prop={'weight': 'bold', 'size': LEGEND_FONTSIZE})

   

    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_save_path = os.path.join(script_dir, out_png_name)

    plt.savefig(full_save_path, dpi=300)

    print(f"成功保存至: {full_save_path}")



# ==========================================

# 第五部分：双数据集对比绘图 (保持原样)

# ==========================================

def get_log_y_setup(values):

    valid_values = [v for v in values if v > 0]

    if not valid_values: return 1, 100, [1, 10, 100]

    y_max_val = max(valid_values)

    y_min_val = min(valid_values)

    top_exponent = math.ceil(math.log10(y_max_val * 1.5))

    top_limit = 10 ** top_exponent

    if y_min_val < 1: bottom_exponent = -1

    elif y_min_val < 10: bottom_exponent = 0

    else: bottom_exponent = math.floor(math.log10(y_min_val * 0.8))

    bottom_limit = 10 ** bottom_exponent

    ticks = [10 ** i for i in range(bottom_exponent, top_exponent + 1)]

    if len(ticks) < 2: ticks = [bottom_limit, top_limit]

    return bottom_limit, top_limit, ticks



def apply_smart_x_ticks(ax, values, manual_range=None):
    if not values: return

    # --- 1. 确定“逻辑”范围 (也就是你想显示的数字范围，如 0 ~ 1.0) ---
    if manual_range:
        tick_min, tick_max = manual_range
    else:
        raw_min = min(values)
        raw_max = max(values)
        tick_min = math.floor(raw_min * 10) / 10.0
        tick_max = math.ceil(raw_max * 10) / 10.0
        # 兜底
        if tick_max <= tick_min: 
            tick_max = tick_min + 0.1

    # --- 2. 定义拉伸系数 (关键步骤) ---
    # 1.02 表示物理轴比逻辑轴长 2%。
    # 这样逻辑上的 1.0 会被放在物理坐标 1.02 的位置。
    STRETCH_FACTOR = 1.02 

    # --- 3. 生成“逻辑”刻度 ---
    # 我们先在 0~1.0 的范围内找好漂亮的刻度位置
    raw_ticks = MaxNLocator(nbins=5).tick_values(tick_min, tick_max)
    
    # 过滤掉超出范围的
    valid_ticks = [t for t in raw_ticks if (t >= tick_min - 1e-9) and (t <= tick_max + 1e-9)]
    
    # 强制包含两端 (tick_min 和 tick_max)
    if not any(abs(t - tick_min) < 1e-9 for t in valid_ticks):
        valid_ticks = [tick_min] + valid_ticks
    if not any(abs(t - tick_max) < 1e-9 for t in valid_ticks):
        valid_ticks.append(tick_max)
    
    valid_ticks = sorted(list(set(valid_ticks)))

    # --- 4. 计算“物理”刻度位置 ---
    # 将逻辑刻度乘以拉伸系数，得到它们在图中实际应该放置的位置
    # 例如：逻辑 1.0 -> 物理 1.02
    physical_ticks = [t * STRETCH_FACTOR for t in valid_ticks]

    # --- 5. 应用设置 ---
    # 设置物理刻度位置
    ax.set_xticks(physical_ticks)
    
    # 设置显示的文字 (使用逻辑值)
    # 强制把最后一个刻度显示为 tick_max (如 "1.0")
    labels = []
    for t in valid_ticks:
        if abs(t) < 1e-9: 
            labels.append("0")
        elif abs(t - tick_max) < 1e-9: 
            # 这里的 tick_max 就是原来的 1.0，但它现在挂在 1.02 的位置上
            labels.append(f"{tick_max:.1f}") # 强制显示 "1.0"
        else:
            labels.append(f"{t:.2f}".rstrip('0').rstrip('.'))
    
    ax.set_xticklabels(labels)

    # --- 6. 设置视野范围 (xlim) ---
    # 视野范围也要按比例拉大，确保右边框(Spine)正好压在最后一个物理刻度(1.02)上
    # 这样 "1.0" 的标签就会正好显示在右边框下面
    phys_min = tick_min * STRETCH_FACTOR
    phys_max = tick_max * STRETCH_FACTOR
    ax.set_xlim(phys_min, phys_max)



def draw_paired_plot(df, dataset_left, dataset_right, out_png_name):

    global SPINE_WIDTH, LINE_WIDTH, AXIS_LABEL_FONTSIZE, AXIS_LABEL_WEIGHT, TICK_LABEL_FONTSIZE, TICK_LABEL_WEIGHT, LEGEND_FONTSIZE, COLORS, MARKERS

    CAT_DB = ["milvus", "weaviate", "qdrant", "vearch", "redis", "elastic", "pgvector"]

    CAT_LIB = ["faiss", "hnswlib", "hnsw+post"]

    def get_category(algo_name):

        name_lower = str(algo_name).lower()

        for db in CAT_DB:

            if db in name_lower: return "Databases"

        for lib in CAT_LIB:

            if lib in name_lower: return "Libraries"

        return "Algorithms"



    TARGET_ALGORITHMS = [

        "rangepq", "digra", "acorn", "irangegraph",

        "hnsw+post", "hnsw", "ivfpq",

        "milvus-hnsw", "milvus-ivfpq", "milvus-gpu-ivfpq", "milvus-gpu-cagra",

        "weaviate-hnsw", "qdrant",

        "vearch-ivfpq", "vearch-hnsw",

        "redis-hnsw", "elasticsearch-hnsw", "pgvector-hnsw",

        "digra_w1-4_s1-6",

        "Faiss-HNSW", "Faiss-IVFPQ", "HNSWLib", "milvus-gpu-ivfflat"

    ]



    df_subset = df[df["dataset"].isin([dataset_left, dataset_right])]

    df_subset = df_subset[df_subset["algorithm"].astype(str).str.lower().isin([x.lower() for x in TARGET_ALGORITHMS])]

    if df_subset.empty: return



    sorted_algos = sorted(df_subset["algorithm"].unique())

    FALLBACK_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    FALLBACK_MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']



    category_configs = [

        {"suffix": "Algorithms", "cat": "Algorithms"},

        {"suffix": "Libraries",  "cat": "Libraries"},

        {"suffix": "Databases",  "cat": "Databases"}

    ]

   

    for config in category_configs:

        target_cat = config["cat"]

        suffix = config["suffix"]

        cat_algos = [a for a in sorted_algos if get_category(a) == target_cat]

        fig, axes = plt.subplots(1, 2, figsize=(40, 18))

        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.71, wspace=0.55)

        row_qps_values = []

        left_recall_values = []

        right_recall_values = []

       

        ax_left = axes[0]

        df_left = df_subset[df_subset["dataset"] == dataset_left]

        for algo in cat_algos:

            algo_data = df_left[df_left["algorithm"] == algo]

            points = list(zip(algo_data["recall"], algo_data["qps"]))

            px, py = create_pareto_frontier(points)

            if not px: continue

            row_qps_values.extend(py)

            left_recall_values.extend(px)

            idx = sorted_algos.index(algo)

            uc = COLORS.get(algo, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])

            um = MARKERS.get(algo, FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)])

            ax_left.plot(list(px), list(py), label=algo, color=uc, marker=um, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9,markevery=MARKER_STEP)



        ax_right = axes[1]

        df_right = df_subset[df_subset["dataset"] == dataset_right]

        for algo in cat_algos:

            algo_data = df_right[df_right["algorithm"] == algo]

            points = list(zip(algo_data["recall"], algo_data["qps"]))

            px, py = create_pareto_frontier(points)

            if not px: continue

            row_qps_values.extend(py)

            right_recall_values.extend(px)

            idx = sorted_algos.index(algo)

            uc = COLORS.get(algo, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])

            um = MARKERS.get(algo, FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)])

            ax_right.plot(list(px), list(py), label=algo, color=uc, marker=um, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9, markevery=MARKER_STEP)



        apply_smart_x_ticks(ax_left, left_recall_values)

        apply_smart_x_ticks(ax_right, right_recall_values)

        final_ylim_bottom, final_ylim_top, final_yticks = get_log_y_setup(row_qps_values)



        for ax, ds_name in zip([ax_left, ax_right], [dataset_left, dataset_right]):

            ax.set_xlabel(f"Recall ({ds_name})", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

            ax.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)

            ax.set_yscale('log')

            ax.set_ylim(bottom=final_ylim_bottom, top=final_ylim_top)

            ax.set_yticks(final_yticks)

            y_formatter = ScalarFormatter(); y_formatter.set_scientific(False)

            ax.yaxis.set_major_formatter(FuncFormatter(smart_y_format))

            ax.yaxis.set_minor_formatter(plt.NullFormatter())

            for spine in ax.spines.values(): spine.set_linewidth(SPINE_WIDTH)

            ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE,width=SPINE_WIDTH,  # 刻度线本身的粗细

               length=10,          # 刻度线长度(可加大)

               pad=20)

            for label in ax.get_xticklabels() + ax.get_yticklabels():

                label.set_fontweight('bold')



        handles_l, labels_l = ax_left.get_legend_handles_labels()

        handles_r, labels_r = ax_right.get_legend_handles_labels()

        all_handles = handles_l + handles_r

        all_labels = labels_l + labels_r



        if all_handles:

            by_label = dict(zip(all_labels, all_handles))

            fig.legend(handles=by_label.values(), labels=by_label.keys(),

                       loc="lower center", bbox_to_anchor=(0.50, 0.84),

                       ncol=4, frameon=False, fontsize=16, prop={'weight': 'bold', 'size': LEGEND_FONTSIZE})



        save_name = out_png_name.replace(".png", f"_{suffix}.png")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        full_save_path = os.path.join(script_dir, save_name)

        plt.savefig(full_save_path, dpi=300)

        plt.close(fig)

        print(f"成功保存独立分类图: {save_name}")



if __name__ == "__main__":

    df_all = parse_all_data()

    if not df_all.empty:

        all_datasets = df_all['dataset'].unique()

        print(f"检测到的所有数据集: {all_datasets}")

       

        for ds in all_datasets:

            if ds and isinstance(ds, str) and len(ds) > 0:

                try:

                    # if ds.startswith("artificial"): continue

                    draw_plot(df_all, ds, f"{ds}.png")

                except Exception as e:

                    print(f"绘制数据集 {ds} 时出错: {e}")

       

        PAIRED_TASKS = [

            ("msong",    "artificial-0.8",  "artificial-80"),

            ("deep1m",   "artificial-0.5",  "artificial-50"),

            ("sift10m",  "artificial-0.06", "artificial-06"),

            ("tiny5m",   "artificial-0.12", "artificial-12")

        ]

       

        print("\n=== 开始绘制双数据集对比图 ===")

        for ds_left, ds_right, alias_right in PAIRED_TASKS:

            if ds_left in all_datasets and ds_right in all_datasets:

                out_name = f"{ds_left}_vs_{alias_right}.png"

                try:

                    draw_paired_plot(df_all, ds_left, ds_right, out_name)

                except Exception as e:

                    print(f"绘制对比图 {out_name} 失败: {e}")

            else:

                print(f"[跳过] 无法找到数据集: {ds_left} 或 {ds_right}")

    else:

        print("未找到数据，请检查文件路径。")