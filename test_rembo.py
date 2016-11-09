# -*- coding: utf-8 -*-

if __name__=='__main__':
    import os
    import time

    try:
        import cPickle as pickle
    except:
        import pickle

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    import hslbo
    from hslbo.example.branin import branin, branin_modified, branin_modified_with_noise

    # Modified Branin function
    func = branin_modified_with_noise

    # 問題の次元
    num_dims = 25

    # 問題の効率的次元
    eff_dims = 2

    # 探索回数(探索点の数)
    num_iter = 500

    # 探索空間の設定
    lower = np.array([-5,0])
    upper = np.array([10,15])

    # MCMCの回数を指定
    acq_options = {"mcmc_iters": 300, "burnin": 3000}

    # シミュレーションの回数
    num_simulation = 20

    # シミュレーションの準備
    total_time = 0

    # 結果を格納するリスト
    results =[]

    for i in range(num_simulation):
        #タイマーをスタート
        start = time.time()

        # BOインスタンスを生成
        bo = hslbo.REMBO(func, num_dims, lower, upper, acq_name="mutual_information", acq_options=acq_options)

        # 最初の1点は初期化した時に探索されるので-1
        bo.sequential_update(num_iter-1)

        # タイマーをストップ
        elapsed_time = time.time() - start
        total_time += elapsed_time

        # 得られた入力の探索点
        inputs = bo.inputs
        # 得られた出力の観測値
        values = bo.values
        # 最適点と最適値の推移
        opt_x_hist, opt_f_hist = bo.history

        # 結果を辞書にまとめる
        result = {"inputs":inputs, "values":values, "opt_x_hist": opt_x_hist, "opt_f_hist": opt_f_hist}

        # 結果をリストに追加
        results.append(result)


    # 結果をオブジェクトとしてそのまま保存
    base = os.path.dirname(__file__)
    print(base)
    name = os.path.join(base, 'result/result01_GPMI_noiseless.pickle')
    with open(name, mode='wb') as f:
        pickle.dump(results,f)

    # 探索にかかった時間の平均
    average_time = total_time / num_simulation
    print("Average computation time of {0} iterations: {1}".format(num_iter, average_time))


    #####################
    ## 結果をプロットする ##
    #####################

    # 最適化の履歴を縦に並べる
    hist = np.array([result["opt_f_hist"] for result in results])

    # 平均と標準偏差をとる
    mean = np.mean(hist,axis=0)
    sigma = np.std(hist, axis=0, ddof=1)

    # 平均と標準偏差をプロット
    x = np.arange(num_iter) + 1
    plt.plot(x, mean, color='b')
    plt.fill_between(x, mean-sigma, mean+sigma, alpha=0.5, color='b')
    # 最適値を赤い水平線でプロット
    MINIMUM_BRANIN_MOD = -15.31007
    plt.axhline(y=MINIMUM_BRANIN_MOD, color='red')

    plt.legend(('GP-MI',))
    plt.xlim(0, num_iter)
    plt.xlabel("Number of iteration")
    plt.ylabel("Minimimum function value")
    plt.title("Modified Branin function")

    plt.savefig('result01_figure.pdf')
    plt.show()


    #グラフのプロットに用いた結果を保存する．
    #np.savez('test.npz', mean=mean, sigma=sigma)
