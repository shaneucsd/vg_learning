import sys
import time

sys.path.append("/Users/apple/Documents/Pycharm/QushiDataPlatform/airflow/dags")
import os
import h5py
import multiprocessing
import numpy as np
import networkx as nx
from tqdm import tqdm
from pyunicorn.timeseries import VisibilityGraph
from ge import Struc2Vec
from postprocess.datakit.DtLoader import DtLoader
from utils.file_utils import load_object_from_pickle

file_lock = multiprocessing.Lock()

DIR = "./data"
TEMP_DIR = "./data/temp"
FILE_PATH = "./data/vg_data.hdf5"


# define enum for different embedding methods, including struc2vec and CI
class EmbeddingMethod:
    STRUC2VEC = "graph_s2v"
    CI = "graph_CI"


def process_graph_vec(args):
    data, period, embed_size, t, n = args
    graph_vec = struc2vec_from_arr_to_graph_vec(data[t - period : t, n], embed_size)
    assert graph_vec is not None
    return graph_vec


def struc2vec_from_arr_to_graph_vec(arr, embed_size=32, window_size=5, i=0):
    # 示例时间序列

    # 创建可见性图对象
    vg = VisibilityGraph(arr)

    # 计算可见性关系
    visibility_matrix = vg.visibility_relations()

    G = nx.from_numpy_array(visibility_matrix)

    model = Struc2Vec(
        G,
        walk_length=10,
        num_walks=80,
        workers=1,
        verbose=0,
        temp_path=TEMP_DIR + f"/temp_struc2vec_{i}/",
    )

    model.train(embed_size=embed_size, window_size=window_size)

    embeddings = np.matrix([model.w2v_model.wv[word] for word in model.graph.nodes()])

    return embeddings


class FeatureData:
    """

    :data , shape = (F, T, N), F: number of features, T: number of time points, N: number of stocks
        data[... , ...] = [open, high, low, close, volume, amount, pct_chg]

    :cld_info = [tradedays, tradeassets]
        tradedays: list of tradedays
        tradeassets: list of tradeassets, int

    :period, int, the number of time points to be considered

    """

    def __init__(self, data: list, cld_info: list, period=20, embed_size=32):
        self.data = data
        self.cld_info = cld_info
        self.period = period
        self.embed_size = embed_size

        for i in range(len(data)):
            assert data[i].shape[1] == data[0].shape[1]

        close = self.data[0]
        self.labels = ((close[1:] - close[:-1]) > 0).astype(int)
        self.myset, self.myset_r = [], []
        T, N = self.data[0].shape

        for t in range(T - 1):
            # skip the last period
            if t < self.period:
                continue
            else:
                for n in range(N):
                    raw = self.data[0][t - self.period : t, n]
                    label = self.labels[t, n]
                    if np.isnan(raw).any() or np.isnan(label):
                        continue
                    else:
                        self.myset.append((t, n))
                        self.myset_r.append((self.cld_info[0][t], self.cld_info[1][n]))
        print(
            f"Valid: {len(self.myset)} samples , kept {(len(self.myset) / (T*N)):.2%} <- Timesteps: {T}, Stocks: {N}, Total: {T*N} samples"
        )
        print(
            f'Example Myset ({self.myset[-1]}): , where represents {self.myset_r[-1]} ', 
        )

    def job_s2v(self, i):
        with file_lock:
            with h5py.File(TEMP_DIR + f"/temp_graph_{i}.hdf5", "w") as f:
                x_s2v_temp = f.create_dataset(
                    f"s2v_temp",
                    shape=(len(self.myset), self.period, self.embed_size),
                    dtype="float32",
                )

                for idx, (t, n) in enumerate(tqdm(self.myset, desc=f"Struc2Vec_{i}")):
                    graph_vec = struc2vec_from_arr_to_graph_vec(
                        self.data[i][t - self.period : t, n], self.embed_size, i=i
                    )
                    try:
                        assert graph_vec is not None
                        x_s2v_temp[idx, ...] = graph_vec
                    except:
                        print(f"Error: {t}, {n}")
                        continue

    def s2v_stack(self):
        G = len(self.data)
        length = len(self.myset)

        temp_files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".hdf5")]

        with h5py.File(os.path.join(TEMP_DIR, temp_files[0]), "r") as f:
            temp_dataset = f["s2v_temp"]
            A, B, C = temp_dataset.shape

        with h5py.File(FILE_PATH, "a") as f:
            combined_dataset = f.create_dataset(
                "s2v", shape=(length, G, self.period, self.embed_size), dtype="float32"
            )
            for i, temp_file in enumerate(
                tqdm(temp_files, desc="Combining s2v temp files")
            ):
                with h5py.File(os.path.join(TEMP_DIR, temp_file), "r") as temp_f:
                    combined_dataset[:, i, ...] = temp_f["s2v_temp"][...]
                os.remove(os.path.join(TEMP_DIR, temp_file))

    def dump(self):
        print("Dumping data to hdf5 file...")
        length = len(self.myset)
        G = len(self.data)

        with h5py.File(FILE_PATH, "w") as f:
            myset = f.create_dataset("myset", shape=(len(self.myset), 2), dtype="int32")
            myset_r = f.create_dataset(
                "myset_r", shape=(len(self.myset_r), 2), dtype="int32"
            )
            x_raw = f.create_dataset(
                "raw", shape=(length, G, self.period), dtype="float32"
            )
            y = f.create_dataset("label", shape=(length,), dtype="int32")

            # save myset, myset_r, x_raw, y
            for idx, (t, n) in enumerate(self.myset):
                myset_r[idx, ...] = [t, n]
                myset[idx, ...] = [t, n]
                y[idx] = self.labels[t, n]

                for i in range(G):
                    x_raw[idx, i, ...] = self.data[i][t - self.period : t, n]
    
    def embed_s2v(self):
        G = len(self.data)
        print('Init embedding...')
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(self.job_s2v, [i for i in range(G)])
        print('Finished embedding')
        
        self.s2v_stack()
        print('Finished Stacking')
        

if __name__ == "__main__":
    dl = DtLoader("stock")
    dtStart = dl.close.index.get_loc('2019-12-02')
    dtEnd =  dl.close.index.get_loc('2023-01-04')
    data = [
        dl.close,
        dl.open,
        dl.high,
        dl.low,
        dl.volume,
        dl.amount,
        dl.vwap,
    ]
    pool = load_object_from_pickle("stock_pool_000300")
    data = [d.where(pool == 1) for d in data]
    data = [d.iloc[dtStart:dtEnd, :] for d in data]
    data = [d.values for d in data]
    # print(data[0].shape)

    cld_info = [
        [int(dt.strftime("%Y%m%d")) for dt in dl.tradedays[dtStart:dtEnd]],
        [int(tradeasset) for tradeasset in dl.tradeassets],
    ]
    del dl

    feature_data = FeatureData(data=data, cld_info=cld_info, period=20, embed_size=32)

    feature_data.dump()
    feature_data.embed_s2v()