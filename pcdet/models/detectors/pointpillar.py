from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # print(self.module_list)

    def forward(self, batch_dict):              # 前向传播
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        if self.training:                       # 训练模式
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict) # 推理模式，返回预测结果和召回值
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()      # 这里对应pointpillar.yaml里的DENSE_HEAD，，根据那里的值确定dense_head模块
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
