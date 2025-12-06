import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="train or plot")
    parser.add_argument('--scenes_file', type=str, 
        help="CloudSat scenes file")
    parser.add_argument('--physical_file',type=str,
        help="Physical file")
    parser.add_argument('--run_name', type=str, default="",
        help="Suffix to use for this training run")
    # 作用：添加另一个可选命令行参数 --run_name，它也是一个字符串（str），表示训练运行的后缀（例如，训练日志文件、模型文件等命名时可以加上这个后缀）。如果没有提供这个参数，默认值是空字符串。
    args = parser.parse_args()
    mode = args.mode
    scenes_fn = args.scenes_file
    physical_fn = args.physical_file

    if mode == "train":
        import train
        train.train_cs_modis_cgan_full(scenes_fn, physical_fn)
    elif mode == "plot":
        import plots
        plots.plot_all(scenes_fn)

# python main.py plot --scenes_file="D:\BR\CGAN-dpi\cloudsat-gan-master\cloudsat-gan-master\cs_modis_scenes.nc" --physical_file="D:\BR\CGAN-dpi\cloudsat-gan-master\cloudsat-gan-master\winducgan_v1\.nc"
