import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
import glob
import math
import matplotlib.pyplot as plt
# ... 其他 import
# 在原本的 import 列表里补充这两行
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import math
from matplotlib.ticker import MultipleLocator, FuncFormatter




from matplotlib.ticker import ScalarFormatter
AXIS_LABEL_FONTSIZE = 75
AXIS_LABEL_WEIGHT   = "bold"
AXIS_LABEL_PAD      = 25   # 新增：让轴标签离刻度远一点，否则会重叠
TICK_LABEL_FONTSIZE = 70
TICK_LABEL_WEIGHT   = "bold"

LEGEND_FONTSIZE     = 70

SUBTITLE_FONTSIZE   = 70
SUBTITLE_WEIGHT     = "bold"
SPINE_WIDTH         = 3.5
LINE_WIDTH          = 4.0














plt.rcParams["font.family"] = "serif"







plt.rcParams["font.serif"] = ["Times New Roman"]















COLORS = {

    "milvus-gpu-ivfflat": "#1f77b4",

    "milvus-hnsw": "#ff7f0e",

    "qdrant": "#9467bd",

    "weaviate-hnsw": "#d62728",

    # --- 您原有的配置 (保持兼容性) ---

    "iRangeGraph": "#d62728",

    "Faiss-HNSW": "#1f77b4",  
    "acorn": "#17becf",              # 改为青色 (Cyan) - 区分度高
    "rangepq": "#e377c2",
    "redis-hnsw": "#8c564b",
    "HNSWLib": "#ff7f0e",    

    "Faiss-IVFPQ": "#2ca02c",

    "results": "#9467bd",

}



MARKERS = {

    "milvus-gpu-ivfflat": "v",

    "milvus-hnsw": "P",

    "qdrant": "o",

    "weaviate-hnsw": "s",

    # --- 您原有的配置 ---

    "iRangeGraph": "o",

    "Faiss-HNSW": "s",  

    "HNSWLib": "^",    

    "Faiss-IVFPQ": "D",

    "results": "X",
    
    "redis-hnsw": "*",

}















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

# 第三部分：数据解析 (自动读取所有 csv/pkl/log)

# ==========================================

def parse_all_data():
    all_data = []
    # 获取脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"正在扫描目录: {script_dir} ...")
    
    # -------------------------------------------------
    # 1. 自动处理所有 .csv, .pkl, .xlsx 文件
    # -------------------------------------------------
    data_files = (
        glob.glob(os.path.join(script_dir, "*.csv")) + 
        glob.glob(os.path.join(script_dir, "*.pkl")) + 
        glob.glob(os.path.join(script_dir, "*.xlsx"))
    )

    if not data_files:
        print("【警告】在该目录下没有找到任何数据文件 (.csv/.pkl/.xlsx)！")

    for fpath in data_files:
        fname = os.path.basename(fpath)
        try:
            # --- 1. Excel 文件处理 ---
            if fname.endswith('.xlsx'):
                df = pd.read_excel(fpath)
                print(f"[成功] 读取Excel: {fname}")

            # --- 2. CSV 文件处理 ---
            elif fname.endswith('.csv'):
                fname_lower = fname.lower()
                is_artificial = "artificial" in fname_lower or "artficial" in fname_lower
                
                if is_artificial:
                    df = pd.read_csv(fpath, header=None)
                    if df.shape[1] >= 4:
                        df.rename(columns={1: 'recall', 2: 'qps'}, inplace=True) # 注意：有时是第2列和第4列(索引1,3)，视情况调整
                        # 如果之前代码是 {1: 'recall', 3: 'qps'} 请保持您确认正确的索引
                        if 'qps' not in df.columns and 3 in df.columns: df.rename(columns={3: 'qps'}, inplace=True)

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
            rename_map = {'k-nn': 'recall', 'knn': 'recall', 'recall@1': 'recall'}
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
                # print(f"[跳过] 文件 {fname} 格式不对") 
                pass
                
        except Exception as e:
            print(f"[错误] 无法读取文件 {fname}: {e}")

    # -------------------------------------------------
    # 2. 解析 Log 文件 (这里是重点修改区域！！！)
    # -------------------------------------------------
    log_files = {
        'Faiss-HNSW': ['faiss_hnsw.log', 'faiss_hnsw_151'],
        'Faiss-IVFPQ': ['faiss_ivfpq_dynamic.log', 'faiss_ivfpq_151'],
        'HNSWLib': ['hnswlib.log', 'bvbj_hnswlib_151'],
        'hnsw+post': ['hnsw_post.log', 'faiss_hnsw_151'],
        
        # 您的 ACORN 配置
        'ACORN': ['acorn.log', '*gamma*', '*acorn*', '*_efc*', '*_efs*', 'output_*'] 
    }

    # 这里的 processed_files 用于防止同一个文件因为匹配多个模式被读两遍
    processed_files = set()

    for algo_name, patterns in log_files.items():
        matched_files = []
        for pattern in patterns:
            # 这里的 pattern + "*" 逻辑是您原有的，虽然有点怪，但能匹配到 acorn.log*
            # 关键是下面的 glob 能否匹配到您的文件
            found = glob.glob(os.path.join(script_dir, pattern + "*")) 
            if not found and os.path.exists(os.path.join(script_dir, pattern)):
                found = [os.path.join(script_dir, pattern)]
            matched_files.extend(found)
        
        # 去重
        matched_files = list(set(matched_files))

        for fpath in matched_files:
            if fpath in processed_files: continue
            
            try:
                fname = os.path.basename(fpath)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                count = 0
                
                # 🔥🔥🔥 分支 A: 标准格式 (包含 [RESULT]) 🔥🔥🔥
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

                # 🔥🔥🔥 分支 B: ACORN 专用格式 (您之前代码里缺了这个！) 🔥🔥🔥
                # 如果文件里没有 [RESULT]，就尝试用 ACORN 的逻辑读
                else:
                    # 1. 尝试从文件内容找 dataset: xxx
                    current_dataset = None
                    ds_match = re.search(r'dataset\s*[:=]\s*([a-zA-Z0-9_\-]+)', content, re.IGNORECASE)
                    if ds_match:
                        current_dataset = ds_match.group(1)
                    else:
                        # 核心补救：如果文件内容里没写 dataset，直接用【文件名】当前缀
                        # 例如 ag_news_N120000... 就会被提取为 ag_news_N120000...
                        # 后续的 normalize_dataset_name 会把它修整为 ag_news
                        current_dataset = os.path.splitext(fname)[0]

                    if current_dataset:
                        for line in lines:
                            # 匹配 QPS=6.4, recall@100=0.9874 这种格式
                            # 注意：正则要允许 QPS 前后可能有空格或逗号
                            qps_match = re.search(r'QPS\s*=\s*([\d\.]+)', line)
                            rec_match = re.search(r'recall@\d+\s*=\s*([\d\.]+)', line)
                            
                            if qps_match and rec_match:
                                all_data.append({
                                    'algorithm': algo_name, # 强制标记为 ACORN
                                    'dataset': current_dataset,
                                    'recall': float(rec_match.group(1)),
                                    'qps': float(qps_match.group(1))
                                })
                                count += 1

                if count > 0:
                    processed_files.add(fpath)
                    print(f"[成功] 读取Log: {fname} (包含 {count} 条数据)")
                    
            except Exception as e:
                # print(f"出错: {e}")
                pass

    # ==========================================
    # 数据集标准化与返回
    # ==========================================
    df_temp = pd.DataFrame(all_data)

    def normalize_dataset_name(name):
        name = str(name).lower()
        if 'artficial' in name or 'artificial' in name:
            if '80' in name or '0.8' in name: return 'artificial-0.8'
            if '50' in name or '0.5' in name: return 'artificial-0.5'
            if '12' in name or '0.12' in name: return 'artificial-0.12'
            if '06' in name or '0.06' in name: return 'artificial-0.06'
            return name
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


def get_log_y_setup(values):
    """
    计算对数坐标 (Log Scale) 的 Y 轴范围和刻度。
    生成如 1, 10, 100, 1000, 10000 这样的刻度。
    注意：Log 轴不能包含 0，如果数据中有 0，将被忽略或视为极小值。
    """
    # 过滤掉 <= 0 的值，因为 log 无法处理 0
    valid_values = [v for v in values if v > 0]
    
    if not valid_values:
        return 1, 100, [1, 10, 100] # 兜底
        
    y_max_val = max(valid_values)
    y_min_val = min(valid_values)
    
    # 1. 计算上限 (Top): 向上取整到最近的 10 的幂
    # 例如 max=450 -> log10=2.65 -> ceil=3 -> 10^3 = 1000
    top_exponent = math.ceil(math.log10(y_max_val * 1.1)) # *1.1 留一点顶部空间
    top_limit = 10 ** top_exponent
    
    # 2. 计算下限 (Bottom): 向下取整到最近的 10 的幂
    # 例如 min=25 -> log10=1.39 -> floor=1 -> 10^1 = 10
    # 如果为了展示更全，通常建议由 1 (10^0) 开始，或者根据数据动态调整
    # 这里我们设定：如果最小值大于10，就从下面一个量级开始；否则从 1 开始
    if y_min_val < 1:
        bottom_exponent = -1 # 0.1
    else:
        bottom_exponent = 0 # 默认为 1
        # 如果数据非常大（比如最小都是 5000），才动态调整下限，否则保持从 1 或 10 开始看起来更直观
        if y_min_val > 100: 
             bottom_exponent = math.floor(math.log10(y_min_val * 0.5))

    bottom_limit = 10 ** bottom_exponent
    
    # 3. 生成刻度：生成 10 的幂次列表
    # range 包含左不包含右，所以要 +1
    ticks = [10 ** i for i in range(bottom_exponent, top_exponent + 1)]
    
    # 修正：如果算出来的 range 太小（比如 min=80, max=90，导致 ticks 只有一个 100），
    # 强制拉大范围以保证至少有两个刻度
    if len(ticks) < 2:
        ticks = [bottom_limit, top_limit]

    return bottom_limit, top_limit, ticks

def apply_smart_x_ticks(ax, values, manual_range=None):
    if not values: return

    # --- 1. 计算端点 (保持你的逻辑) ---
    if manual_range:
        tick_min, tick_max = manual_range
    else:
        raw_min = min(values)
        raw_max = max(values)
        # 向下/向上取整到一位小数
        tick_min = math.floor(raw_min * 10) / 10.0
        tick_max = math.ceil(raw_max * 10) / 10.0

        if tick_max <= tick_min:
            tick_max = tick_min + 0.1

    # --- 2. 关键修改：只用 Locator，不要用 set_xticks ---
    
    # 设置视图范围 (这将保证左右端点是 0.2 和 0.8 这种整位)
    ax.set_xlim(tick_min, tick_max)
    
    # 设置刻度规则：严格每隔 0.1 一个
    # Matplotlib 会自动从 tick_min 开始算，所以必定会包含端点
    ticks = np.linspace(tick_min, tick_max, 5)
    
    # 强制设置这5个刻度
    ax.set_xticks(ticks)

    # --- 3. 格式化显示 ---
    def custom_fmt(x, pos):
        if abs(x) < 1e-9: return "0"
        if abs(x - 1.0) < 1e-9: return "1.0"
        return f"{x:.2g}"
        
    ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
# ==========================================

# 第四部分：绘图主函数 (修改版：支持自动颜色)

# ==========================================

def draw_plot(df, dataset_name, out_png_name):

    # ==========================================

    # 【新增】算法白名单过滤 (只保留表格中指定的算法)

    # ==========================================

    TARGET_ALGORITHMS = [

        "RangePQ", "DiGRA", "ACORN", "iRangeGraph",

        "hnsw+post", "hnsw", "ivfpq",

        "milvus-hnsw", "milvus-ivfpq", "milvus-gpu-ivfpq", "milvus-gpu-cagra",

        "weaviate-hnsw", "qdrant",

        "vearch-ivfpq", "vearch-hnsw",

        "redis-hnsw", "elasticsearch-hnsw", "pgvector-hnsw",

        "digra_w1-4_s1-6",

        "FAISS-hnsw",       # 对应表格中的 hnsw (faiss版)

        "FAISS-ivfpq",      # 对应表格中的 ivfpq (faiss版)

        "HNSWlib",          # 对应表格中的 hnswlib

        "milvus-gpu-ivfflat" 
    ]
    df_temp = df[df["dataset"] == dataset_name]

   

    # 🔥🔥🔥【诊断代码开始】🔥🔥🔥

    # 如果是 ag_news，我们打印一下到底读到了哪些原始算法名

    if dataset_name == "artificial-0.8":  

        raw_algos = df_temp["algorithm"].unique()

        print(f"\n[诊断报告] artificial-0.8 原始读取到的算法 ({len(raw_algos)}个):")

        print(raw_algos)

        print(f"[诊断报告] 白名单列表:")

        print(TARGET_ALGORITHMS)

       

        ignored = [algo for algo in raw_algos if str(algo).lower() not in [t.lower() for t in TARGET_ALGORITHMS]]

        if ignored:

            print(f"⚠️ [警告] 以下算法被白名单过滤掉了 (名字不匹配?): {ignored}")

            print("👉 请将这些名字手动添加到 TARGET_ALGORITHMS 列表中！")

        else:

            print("✅ 没有算法因为名字问题被过滤。")

    # 🔥🔥🔥【诊断代码结束】🔥🔥🔥

    # 注意：这里做一次不区分大小写的匹配，防止大小写差异导致漏画

    # 这一步会把 df 中不在列表里的算法剔除

    df = df[df["algorithm"].astype(str).str.lower().isin([x.lower() for x in TARGET_ALGORITHMS])]



    # ... (原有代码: df_subset = df[df["dataset"] == dataset_name] ...) ...

    df_subset = df[df["dataset"] == dataset_name]

    if df_subset.empty: return

   

    # ... (后续代码不变) ...

    # === 新增代码开始：定义分类 ===

    CAT_DB = ["milvus", "weaviate", "qdrant", "vearch", "redis", "elastic", "pgvector"]

    CAT_LIB = ["faiss", "hnswlib"]
    def get_category(algo_name):

        name_lower = str(algo_name).lower()

        for db in CAT_DB:

            if db in name_lower: return "Databases" # 数据库

        for lib in CAT_LIB:

            if lib in name_lower: return "Libraries" # 算法库

        return "Algorithms" # 纯算法

    # === 新增代码结束 ===

    df_subset = df[df["dataset"] == dataset_name]

    if df_subset.empty: return



    algorithms = df_subset["algorithm"].unique()

    print(f"正在绘制 {dataset_name} (包含: {algorithms})")

   

    # 排序：iRange 优先，其他的按字母顺序

    sorted_algos = sorted(algorithms)

    # === 新增代码开始：定义特殊数据集 ===

    SPECIAL_DATASETS = ["ag_news", "app_reviews", "amazon", "cc_news"]

    is_special_task = dataset_name in SPECIAL_DATASETS

    # === 新增代码结束 ===



    # --- 新增：定义备用的颜色和形状池 (用于未在 COLORS 字典中定义的算法) ---

    # 使用 matplotlib 默认的十色环，避开灰色

    FALLBACK_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 定义一堆可选的形状

    FALLBACK_MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']



    # === 核心修改开始 ===

    # === 核心修改开始 ===
    if is_special_task:
        # 【情景 A】特殊数据集：画 1 行 3 列
        # 1. 修改画布：高度从 6 改为 11，给巨大的图例留出空间
        fig, axes = plt.subplots(1, 3, figsize=(60, 15)) 

        # 2. 初始化图例收集列表
        all_handles = []
        all_labels = []

        # 3. 调整间距
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.61, wspace=0.55)
        
        # 定义三个子图的内容
        plot_configs = [
            {"title": "(a) Algorithms", "cat": "Algorithms"},
            {"title": "(b) Libraries",  "cat": "Libraries"},
            {"title": "(c) Databases",  "cat": "Databases"}
        ]

        # 循环 3 次，画 3 个子图
        for idx, config in enumerate(plot_configs):
            ax = axes[idx]
            target_cat = config["cat"]
            
            # --- 初始化数据收集列表 ---
            ax_x_values = []
            ax_y_values = []
            
            # 只筛选当前类别的算法
            current_algos = [a for a in sorted_algos if get_category(a) == target_cat]
            
            for i, algo in enumerate(current_algos):
                algo_data = df_subset[df_subset["algorithm"] == algo]
                points = list(zip(algo_data["recall"], algo_data["qps"]))
                px, py = create_pareto_frontier(points)
                if not px: continue
                px, py = list(px), list(py)
                
                # 颜色和 Marker
                if algo in COLORS: use_color = COLORS[algo]
                else: use_color = FALLBACK_COLORS[sorted_algos.index(algo) % len(FALLBACK_COLORS)]
                
                if algo in MARKERS: use_marker = MARKERS[algo]
                else: use_marker = FALLBACK_MARKERS[sorted_algos.index(algo) % len(FALLBACK_MARKERS)]

                # 绘图
                ax.plot(px, py, label=algo, color=use_color, marker=use_marker,
                        linewidth=LINE_WIDTH, markersize=18, alpha=0.9)
                
                # --- 收集数据 ---
                ax_x_values.extend(px)
                ax_y_values.extend(py)

            # ===============================================
            # 🔥🔥🔥 修改点：应用对数 Y 轴 🔥🔥🔥
            # ===============================================
            
            # 1. 开启 Log 模式 (必须最先设置)
            ax.set_yscale('log')
            
            # 2. 应用智能 Y 轴 (使用新函数 get_log_y_setup)
            if ax_y_values:
                y_bottom, y_top, y_ticks = get_log_y_setup(ax_y_values)
                ax.set_ylim(y_bottom, y_top)
                ax.set_yticks(y_ticks)
                
                # 3. 格式化数字：显示 1, 10, 100 而不是 10^0
                y_formatter = ScalarFormatter(); y_formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(y_formatter)
                ax.yaxis.set_minor_formatter(plt.NullFormatter()) # 隐藏次级小刻度
                
                ax.tick_params(axis='both', which='major', pad=40, width=SPINE_WIDTH, length=15)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight(TICK_LABEL_WEIGHT)
                    label.set_fontsize(TICK_LABEL_FONTSIZE)
            
            # 3. 应用智能 X 轴 (保持不变)
            if ax_x_values:
                apply_smart_x_ticks(ax, ax_x_values)
                ax.tick_params(axis='both', which='major', pad=40, width=SPINE_WIDTH, length=15)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight(TICK_LABEL_WEIGHT)
                    label.set_fontsize(TICK_LABEL_FONTSIZE)

            # 设置标签
            ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)
            ax.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT) # 提示这是 Log Scale
            
            ax.set_title(config["title"], y=-0.5, fontsize=SUBTITLE_FONTSIZE, fontweight=SUBTITLE_WEIGHT)
            
            # 加粗边框
            for spine in ax.spines.values(): 
                spine.set_linewidth(SPINE_WIDTH)
            
            # 收集图例
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
        
        # === 统一大图例 ===
        if all_handles:
            by_label = dict(zip(all_labels, all_handles))
            legend_font = {'size': LEGEND_FONTSIZE, 'weight': 'bold'}
            fig.legend(handles=by_label.values(), labels=by_label.keys(), 
                       loc="upper center",    
                       bbox_to_anchor=(0.5, 1.0), 
                       ncol=5,
                       frameon=False, 
                       prop=legend_font,
                       columnspacing=1.2,
                       labelspacing=0.3)

    else:
        # 【情景 B】普通数据集
        fig, ax = plt.subplots(figsize=(6.6, 5.9))
        plt.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.85)
        
        ax_x_values = []
        ax_y_values = []

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
                    linewidth=LINE_WIDTH, markersize=8, alpha=0.9)
            
            ax_x_values.extend(px)
            ax_y_values.extend(py)

        # ===============================================
        # 🔥🔥🔥 修改点：应用对数 Y 轴 🔥🔥🔥
        # ===============================================
        
        # 1. 开启 Log 模式
        ax.set_yscale('log')
        
        # 2. 应用智能 Y 轴
        if ax_y_values:
            y_bottom, y_top, y_ticks = get_log_y_setup(ax_y_values)
            ax.set_ylim(y_bottom, y_top)
            ax.set_yticks(y_ticks)
            
            # 格式化数字
            y_formatter = ScalarFormatter(); y_formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

        # 3. 应用智能 X 轴
        if ax_x_values:
            apply_smart_x_ticks(ax, ax_x_values)

        ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)
        ax.set_ylabel("QPS (Log Scale)", fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT)
        
        ax.tick_params(axis='both', which='major', pad=40, width=SPINE_WIDTH, length=15)
        for spine in ax.spines.values(): spine.set_linewidth(SPINE_WIDTH)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(TICK_LABEL_WEIGHT)
            label.set_fontsize(TICK_LABEL_FONTSIZE)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_items = len(handles)
            final_ncol = min(n_items, 3) if n_items > 0 else 1
            legend_font = {'size': LEGEND_FONTSIZE, 'weight': 'bold'}
            ax.legend(handles=handles, labels=labels, loc="lower center",
                      bbox_to_anchor=(0.5, 1.00), borderaxespad=0.4,
                      ncol=6, prop=legend_font, columnspacing=1.0)

    # === 核心修改结束 ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(script_dir, out_png_name)
    plt.savefig(full_save_path, dpi=600)

    print(f"成功保存至: {full_save_path}")
# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 读取数据
    df_all = parse_all_data()

    if not df_all.empty:
        # 获取数据中出现过的所有数据集名称（自动去重）
        all_datasets = df_all['dataset'].unique()
        
        print(f"检测到的所有数据集: {all_datasets}")
        
        # 2. 遍历所有检测到的数据集进行绘图
        for ds in all_datasets:
            # 过滤掉非法的空名字
            if ds and isinstance(ds, str) and len(ds) > 0:
                try:
                    # 调用画图函数
                    draw_plot(df_all, ds, f"{ds}.png")
                except Exception as e:
                    # 打印错误但不中断程序
                    import traceback
                    print(f"绘制数据集 {ds} 时出错: {e}")
                    traceback.print_exc()
    else:
        print("未找到数据，请检查文件路径。")