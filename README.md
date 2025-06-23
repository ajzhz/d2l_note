# dive into deep leraning note
`conda env create -f environment.yml`一键配置环境

## 常见问题
1. 我安装的是`1.0.3`版本的d2l，其中有很多实现和课程网站上的调用不一样，例如transformer中的一堆num_hidden在我的版本中全部优化为一个变量，在使用时还是需要查看一下自己的版本中的实现是怎么样的。还有一部分函数在该版本中被删除，主要集中在RNN章节，如果你发现`d2l.torch`中没有你调用的函数，那么你只需要将你**之前课程**中实现的函数复制到`d2l.torch`中，重启内核(jupyter)就可以正常使用了。
2. 我在运行课程后期的所有dataloader类中全部卡死，运行了`很长时间都没有任何反应`，我的判断是windows的多线程有问题，如果你也碰到了这个问题，我的解决方法供你参考：将代码中所有的 `num_worker = d2l.get_dataloader_workers()`改为`num_worker = 0`，如果这个问题那么大概率你在最后的bert微调时也会卡死，记得在`SNLIBERTDataset`这个类中将_preprocess函数改为下面的形式
```
    def _preprocess(self, all_premise_hypothesis_tokens):
        out = [self._mp_worker(pair) for pair in all_premise_hypothesis_tokens] 
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))
```
3. 在`数据集下载时也会报错`问题出在这个函数`d2l.download_extract`他的很多网址无法下载或者无法连接，这时你需要将数据手动下载到../data下，并自行解压，之后将`data_dir = d2l.download_extract(..)`这行代码改为`data_dir = ../data/你要训练的数据集文件名称`就可以进行下一步了
   