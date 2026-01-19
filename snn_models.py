import torch
import torch.nn as nn
import snntorch as snn



class mtsnnMLP(torch.nn.Module):
    def __init__(self, ninputs, nhidden, noutputs, num_steps=4, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        # 网络层
        self.dense1 = torch.nn.Linear(ninputs, nhidden)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.dense2 = torch.nn.Linear(nhidden, noutputs)
        self.lif2 = snn.Leaky(beta=self.beta, output=True)  # 输出层的LIF神经元
        
    def forward(self, x):
        # 初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        # 记录输出脉冲和膜电位
        spk_rec = []
        mem_rec = []
        # 时间步循环
        for step in range(self.num_steps):
            # 第一层：全连接 + LIF
            cur_x = self.dense1(x)
            spk1, mem1 = self.lif1(cur_x, mem1)
            
            # 第二层：全连接 + LIF
            cur_x = self.dense2(spk1)
            spk2, mem2 = self.lif2(cur_x, mem2)
            
            # 记录输出
            spk_rec.append(spk2)
            mem_rec.append(mem2)
        
        # 返回所有时间步的脉冲（用于spike rate编码）
        #spk_out = torch.stack(spk_rec, dim=0)  # [T, B, N]
        
        #返回平均膜电位:更稳定的分类
        mem_out = torch.stack(mem_rec, dim=0).mean(dim=0)  # [B, N]
        
        #返回脉冲计数
        #spike_count = torch.stack(spk_rec, dim=0).sum(dim=0)  # [B, N]
        
        return mem_out  # spk_out 或 spike_count



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, cfg, num_classes=10, input_size=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        
        if cfg is None:
            self.loss_name = 'unimodel'
            self.beta = 0.9
            self.num_steps = 10
            dropout_rate = 0.5
        else:
            self.loss_name = cfg.loss.name if hasattr(cfg.loss, 'name') else cfg.loss
            dropout_rate = cfg.model.layer.dropout_rate if hasattr(cfg.model.layer, 'dropout_rate') else 0.5
        


        # Check if using ordinal loss that requires special output
        self.loss_name = cfg.loss.name if hasattr(cfg.loss, 'name') else cfg.loss
        
        # Determine number of output neurons based on loss function
        if self.loss_name in ['OrdinalEncoding', 'ORD_ACL', 'VS_SL', 'NeuronStickBreaking']:
            self.num_outputs = num_classes - 1
        elif self.loss_name in ['POM', 'MAE', 'MSE']:
            self.num_outputs = 1
        else:
            self.num_outputs = num_classes
            
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_outputs)
        
        # Add dropout if specified in config
        dropout_rate = cfg.model.layer.dropout_rate if hasattr(cfg.model.layer, 'dropout_rate') else 0.5
        self.dropout = nn.Dropout(dropout_rate)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, save_maps=False):
        # save_maps parameter for compatibility with FlexibleSNN interface
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 使用LIF神经元的SNN-ResNet18
class SNNBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        out = self.bn1(self.conv1(x))
        out, mem1 = self.lif1(out, mem1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out, mem2 = self.lif2(out, mem2)
        return out

class SNNResNet18(nn.Module):
    def __init__(self, cfg, num_classes=10, input_size=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # Check if using ordinal loss that requires special output
        self.loss_name = cfg.loss.name if hasattr(cfg.loss, 'name') else cfg.loss
        
        # Determine number of output neurons based on loss function
        if self.loss_name in ['OrdinalEncoding', 'ORD_ACL', 'VS_SL', 'NeuronStickBreaking']:
            self.num_outputs = num_classes - 1
        elif self.loss_name in ['POM', 'MAE', 'MSE']:
            self.num_outputs = 1
        else:
            self.num_outputs = num_classes
        
        # Get SNN specific parameters from config
        self.beta = cfg.model.neuron.beta if hasattr(cfg.model.neuron, 'beta') else 0.9
        self.num_steps = cfg.model.time_steps if hasattr(cfg.model, 'time_steps') else 10
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1, self.beta)
        self.layer2 = self._make_layer(64, 128, 2, 2, self.beta)
        self.layer3 = self._make_layer(128, 256, 2, 2, self.beta)
        self.layer4 = self._make_layer(256, 512, 2, 2, self.beta)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_outputs)
        self.lif_out = snn.Leaky(beta=self.beta, output=True)
        
        # Add dropout if specified in config
        dropout_rate = cfg.model.layer.dropout_rate if hasattr(cfg.model.layer, 'dropout_rate') else 0.5
        self.dropout = nn.Dropout(dropout_rate)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, beta):
        layers = []
        layers.append(SNNBasicBlock(in_channels, out_channels, stride, beta))
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(out_channels, out_channels, beta=beta))
        return nn.Sequential(*layers)
    
    def forward(self, x, save_maps=False):
        # save_maps parameter for compatibility with FlexibleSNN interface
        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        
        for step in range(self.num_steps):
            cur_x = x
            cur_x = self.bn1(self.conv1(cur_x))
            cur_x, mem1 = self.lif1(cur_x, mem1)
            cur_x = self.maxpool(cur_x)
            
            cur_x = self.layer1(cur_x)
            cur_x = self.layer2(cur_x)
            cur_x = self.layer3(cur_x)
            cur_x = self.layer4(cur_x)
            
            cur_x = self.avgpool(cur_x)
            cur_x = torch.flatten(cur_x, 1)
            cur_x = self.dropout(cur_x)
            cur_x = self.fc(cur_x)
            
            spk, mem_out = self.lif_out(cur_x, mem_out)
            spk_rec.append(spk)
            mem_rec.append(mem_out)
        
        # Return the average membrane potential over time as the output
        # This is more stable for classification tasks
        output = torch.stack(mem_rec, dim=0).mean(dim=0)
        #output = cur_x
        return output




class stats_mtsnnMLP(torch.nn.Module):
    def __init__(self, ninputs, nhidden, noutputs, num_steps=4, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta

        self.dense1 = torch.nn.Linear(ninputs, nhidden)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.dense2 = torch.nn.Linear(nhidden, noutputs)
        self.lif2 = snn.Leaky(beta=self.beta, output=True)  
        
    def forward(self, x, debug=False):
        if debug:
            print(f"\n{'='*60}")
            print(f"MLP Forward Pass Started")
            print(f"{'='*60}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

        # initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []
        mem_rec = []
        lif1_spk_rec = []
        lif2_spk_rec = []


        for step in range(self.num_steps):
            if debug:
                print(f"\n{'-'*60}")
                print(f"Time Step {step + 1}/{self.num_steps}")
                print(f"{'-'*60}")

            cur_x = self.dense1(x)
            if debug:
                print(f"After dense1: shape={cur_x.shape}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}], "
                      f"mean={cur_x.mean():.4f}, std={cur_x.std():.4f}")

            spk1, mem1 = self.lif1(cur_x, mem1)
            lif1_spk_rec.append(spk1)
            if debug:
                spike_rate = spk1.mean().item()
                nonzero_ratio = (spk1 > 0).float().mean().item()
                print(f"After lif1 (spike): shape={spk1.shape}, "
                      f"spike_rate={spike_rate:.4f}, "
                      f"nonzero_ratio={nonzero_ratio:.4f}, "
                      f"range=[{spk1.min():.4f}, {spk1.max():.4f}]")


            cur_x = self.dense2(spk1)
            if debug:
                print(f"After dense2: shape={cur_x.shape}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}], "
                      f"mean={cur_x.mean():.4f}")

            spk2, mem2 = self.lif2(cur_x, mem2)
            lif2_spk_rec.append(spk2)


            spk_rec.append(spk2)
            mem_rec.append(mem2)

            if debug:
                spike_rate = spk2.mean().item()
                print(f"After lif2 (spike): shape={spk2.shape}, spike_rate={spike_rate:.4f}")
                print(f"Membrane potential: range=[{mem2.min():.4f}, {mem2.max():.4f}], "
                      f"mean={mem2.mean():.4f}")


        lif1_spikes = torch.stack(lif1_spk_rec, dim=0)  # [T, B, H] T=time steps, B=batch, H=hidden neurons
        lif2_spikes = torch.stack(lif2_spk_rec, dim=0)  # [T, B, N]

        if debug:
            print(f"\n{'='*60}")
            print(f"Computing Spike Frequency Statistics")
            print(f"{'='*60}")
            print(f"lif1_spikes shape: {lif1_spikes.shape}")
            print(f"  -> T={lif1_spikes.shape[0]} (time steps)")
            print(f"  -> B={lif1_spikes.shape[1]} (batch)")
            print(f"  -> H={lif1_spikes.shape[2]} (hidden neurons)")
            print(f"lif2_spikes shape: {lif2_spikes.shape}")

        # calculate spike frequencies
        self.lif1_spike_frequency = lif1_spikes.mean(dim=0)  # [B, H] average over time steps
        lif1_freq = lif1_spikes.mean().item()
        lif2_freq = lif2_spikes.mean().item()


        lif1_neurons = lif1_spikes.shape[2]
        lif2_neurons = lif2_spikes.shape[2]
        total_neurons = lif1_neurons + lif2_neurons
        overall_frequency = (lif1_freq * lif1_neurons + lif2_freq * lif2_neurons) / total_neurons

   
        self.spike_stats = {
            'lif1': {'frequency': lif1_freq, 'neurons': lif1_neurons, 'shape': lif1_spikes.shape},
            'lif2': {'frequency': lif2_freq, 'neurons': lif2_neurons, 'shape': lif2_spikes.shape},
            'total_neurons': total_neurons,
            'overall_frequency': overall_frequency
        }

        if debug:
            print(f"\n{'Layer':<12} {'Neurons':<12} {'Frequency':<12}")
            print(f"{'-'*36}")
            print(f"{'lif1':<12} {lif1_neurons:<12} {lif1_freq:<12.6f}")
            print(f"{'lif2':<12} {lif2_neurons:<12} {lif2_freq:<12.6f}")
            print(f"{'-'*36}")
            print(f"{'Total':<12} {total_neurons:<12} {overall_frequency:<12.6f}")
            print(f"{'='*60}\n")


        mem_out = torch.stack(mem_rec, dim=0).mean(dim=0)  # [B, N]

        return mem_out, overall_frequency

    def get_lif1_spike_frequency(self):
        """get lif1 layer spike frequency tensor"""
        return self.lif1_spike_frequency

    def get_lif1_frequency_stats(self):
        """get lif1 layer spike frequency statistics"""
        if self.lif1_spike_frequency is None:
            return None

        stats = {
            'mean_frequency': self.lif1_spike_frequency.mean().item(),  # average frequency
            'max_frequency': self.lif1_spike_frequency.max().item(),    # maximum frequency
            'min_frequency': self.lif1_spike_frequency.min().item(),    # minimum frequency
            'std_frequency': self.lif1_spike_frequency.std().item(),    # standard deviation
            'frequency_per_neuron': self.lif1_spike_frequency.mean(dim=0)  # frequency per neuron
        }
        return stats

    def get_all_spike_stats(self):
        """get all layers spike statistics"""
        if not hasattr(self, 'spike_stats'):
            return None
        return self.spike_stats

    def print_spike_stats(self):
        """print formatted spike statistics"""
        if not hasattr(self, 'spike_stats'):
            print("No spike statistics available. Run forward pass first.")
            return

        print(f"\n{'='*70}")
        print(f"Spike Firing Rate Statistics for MLP (All Layers)")
        print(f"{'='*70}")
        print(f"{'Layer':<12} {'Neurons':<15} {'Frequency':<15} {'Contribution':<15}")
        print(f"{'-'*70}")

        total_neurons = self.spike_stats['total_neurons']
        for layer_name in ['lif1', 'lif2']:
            stats = self.spike_stats[layer_name]
            neurons = stats['neurons']
            freq = stats['frequency']
            contribution = (neurons / total_neurons) * 100
            print(f"{layer_name:<12} {neurons:<15} {freq:<15.6f} {contribution:<15.2f}%")

        print(f"{'-'*70}")
        print(f"{'Total':<12} {total_neurons:<15} {self.spike_stats['overall_frequency']:<15.6f} {'100.00%':<15}")
        print(f"{'='*70}\n")




class stats_SNNResNet18(nn.Module):
    def __init__(self, cfg, num_classes=10, input_size=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # Check if using ordinal loss that requires special output
        self.loss_name = cfg.loss.name if hasattr(cfg.loss, 'name') else cfg.loss
        
        # Determine number of output neurons based on loss function
        if self.loss_name in ['OrdinalEncoding', 'ORD_ACL', 'VS_SL', 'NeuronStickBreaking']:
            self.num_outputs = num_classes - 1
        elif self.loss_name in ['POM', 'MAE', 'MSE']:
            self.num_outputs = 1
        else:
            self.num_outputs = num_classes
        
        # Get SNN specific parameters from config
        self.beta = cfg.model.neuron.beta if hasattr(cfg.model.neuron, 'beta') else 0.9
        self.num_steps = cfg.model.time_steps if hasattr(cfg.model, 'time_steps') else 10
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1, self.beta)
        self.layer2 = self._make_layer(64, 128, 2, 2, self.beta)
        self.layer3 = self._make_layer(128, 256, 2, 2, self.beta)
        self.layer4 = self._make_layer(256, 512, 2, 2, self.beta)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_outputs)
        self.lif_out = snn.Leaky(beta=self.beta, output=True)
        
        # Add dropout if specified in config
        dropout_rate = cfg.model.layer.dropout_rate if hasattr(cfg.model.layer, 'dropout_rate') else 0.5
        self.dropout = nn.Dropout(dropout_rate)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, beta):
        layers = []
        layers.append(SNNBasicBlock(in_channels, out_channels, stride, beta))
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(out_channels, out_channels, beta=beta))
        return nn.Sequential(*layers)
    
    def forward(self, x, save_maps=False, debug=False):
        # save_maps parameter for compatibility with FlexibleSNN interface
        # debug parameter to enable detailed printing

        if debug:
            print(f"\n{'='*60}")
            print(f"Forward Pass Started")
            print(f"{'='*60}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        mem_rec = []


        lif1_spk_rec = []
        layer1_spk_rec = []
        layer2_spk_rec = []
        layer3_spk_rec = []
        layer4_spk_rec = []
        lif_out_spk_rec = []

        for step in range(self.num_steps):
            if debug:
                print(f"\n{'-'*60}")
                print(f"Time Step {step + 1}/{self.num_steps}")
                print(f"{'-'*60}")

            cur_x = x

            # Conv1 + BN1
            cur_x = self.bn1(self.conv1(cur_x))
            if debug:
                print(f"After conv1+bn1: shape={cur_x.shape}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}], "
                      f"mean={cur_x.mean():.4f}, std={cur_x.std():.4f}")

            # LIF1
            cur_x, mem1 = self.lif1(cur_x, mem1)
            lif1_spk_rec.append(cur_x)
            if debug:
                spike_rate = cur_x.mean().item()
                nonzero_ratio = (cur_x > 0).float().mean().item()
                print(f"After lif1 (spike): shape={cur_x.shape}, "
                      f"spike_rate={spike_rate:.4f}, "
                      f"nonzero_ratio={nonzero_ratio:.4f}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}]")

            # Maxpool
            cur_x = self.maxpool(cur_x)
            if debug:
                print(f"After maxpool: shape={cur_x.shape}")

            # Layer1
            cur_x = self.layer1(cur_x)
            layer1_spk_rec.append(cur_x)
            if debug:
                spike_rate = cur_x.mean().item()
                print(f"After layer1: shape={cur_x.shape}, spike_rate={spike_rate:.4f}")

            # Layer2
            cur_x = self.layer2(cur_x)
            layer2_spk_rec.append(cur_x)
            if debug:
                spike_rate = cur_x.mean().item()
                print(f"After layer2: shape={cur_x.shape}, spike_rate={spike_rate:.4f}")

            # Layer3
            cur_x = self.layer3(cur_x)
            layer3_spk_rec.append(cur_x)
            if debug:
                spike_rate = cur_x.mean().item()
                print(f"After layer3: shape={cur_x.shape}, spike_rate={spike_rate:.4f}")

            # Layer4
            cur_x = self.layer4(cur_x)
            layer4_spk_rec.append(cur_x)
            if debug:
                spike_rate = cur_x.mean().item()
                print(f"After layer4: shape={cur_x.shape}, spike_rate={spike_rate:.4f}")

            # Avgpool + Flatten
            cur_x = self.avgpool(cur_x)
            cur_x = torch.flatten(cur_x, 1)
            if debug:
                print(f"After avgpool+flatten: shape={cur_x.shape}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}]")

            # Dropout + FC
            cur_x = self.dropout(cur_x)
            cur_x = self.fc(cur_x)
            if debug:
                print(f"After dropout+fc: shape={cur_x.shape}, "
                      f"range=[{cur_x.min():.4f}, {cur_x.max():.4f}], "
                      f"mean={cur_x.mean():.4f}")

            # Output LIF
            spk, mem_out = self.lif_out(cur_x, mem_out)
            lif_out_spk_rec.append(spk)
            spk_rec.append(spk)
            mem_rec.append(mem_out)
            if debug:
                spike_rate = spk.mean().item()
                print(f"After lif_out (spike): shape={spk.shape}, spike_rate={spike_rate:.4f}")
                print(f"Membrane potential: range=[{mem_out.min():.4f}, {mem_out.max():.4f}], "
                      f"mean={mem_out.mean():.4f}")


        lif1_spikes = torch.stack(lif1_spk_rec, dim=0)  # [T, B, C, H, W]
        layer1_spikes = torch.stack(layer1_spk_rec, dim=0)  # [T, B, C, H, W]
        layer2_spikes = torch.stack(layer2_spk_rec, dim=0)  # [T, B, C, H, W]
        layer3_spikes = torch.stack(layer3_spk_rec, dim=0)  # [T, B, C, H, W]
        layer4_spikes = torch.stack(layer4_spk_rec, dim=0)  # [T, B, C, H, W]
        lif_out_spikes = torch.stack(lif_out_spk_rec, dim=0)  # [T, B, N]

        # calculate spike frequencies for each layer
        lif1_freq = lif1_spikes.mean().item()
        layer1_freq = layer1_spikes.mean().item()
        layer2_freq = layer2_spikes.mean().item()
        layer3_freq = layer3_spikes.mean().item()
        layer4_freq = layer4_spikes.mean().item()
        lif_out_freq = lif_out_spikes.mean().item()

        # 
        def count_neurons(spikes):

            # spikes shape: [T, B, ...]

            total_elements = spikes.numel()
            T, B = spikes.shape[0], spikes.shape[1]
            return total_elements // (T * B)

        lif1_neurons = count_neurons(lif1_spikes)
        layer1_neurons = count_neurons(layer1_spikes)
        layer2_neurons = count_neurons(layer2_spikes)
        layer3_neurons = count_neurons(layer3_spikes)
        layer4_neurons = count_neurons(layer4_spikes)
        lif_out_neurons = count_neurons(lif_out_spikes)

        total_neurons = (lif1_neurons + layer1_neurons + layer2_neurons +
                        layer3_neurons + layer4_neurons + lif_out_neurons)


        overall_frequency = (
            lif1_freq * lif1_neurons +
            layer1_freq * layer1_neurons +
            layer2_freq * layer2_neurons +
            layer3_freq * layer3_neurons +
            layer4_freq * layer4_neurons +
            lif_out_freq * lif_out_neurons
        ) / total_neurons


        self.spike_stats = {
            'lif1': {'frequency': lif1_freq, 'neurons': lif1_neurons, 'shape': lif1_spikes.shape},
            'layer1': {'frequency': layer1_freq, 'neurons': layer1_neurons, 'shape': layer1_spikes.shape},
            'layer2': {'frequency': layer2_freq, 'neurons': layer2_neurons, 'shape': layer2_spikes.shape},
            'layer3': {'frequency': layer3_freq, 'neurons': layer3_neurons, 'shape': layer3_spikes.shape},
            'layer4': {'frequency': layer4_freq, 'neurons': layer4_neurons, 'shape': layer4_spikes.shape},
            'lif_out': {'frequency': lif_out_freq, 'neurons': lif_out_neurons, 'shape': lif_out_spikes.shape},
            'total_neurons': total_neurons,
            'overall_frequency': overall_frequency
        }


        self.lif1_spike_frequency = lif1_spikes.mean(dim=0)  # [B, C, H, W]

        if debug:
            print(f"\n{'='*60}")
            print(f"Computing Spike Frequency Statistics (All Layers)")
            print(f"{'='*60}")
            print(f"\n{'Layer':<12} {'Neurons':<12} {'Frequency':<12} {'Shape'}")
            print(f"{'-'*60}")
            for layer_name in ['lif1', 'layer1', 'layer2', 'layer3', 'layer4', 'lif_out']:
                stats = self.spike_stats[layer_name]
                print(f"{layer_name:<12} {stats['neurons']:<12} {stats['frequency']:<12.6f} {str(stats['shape'])}")
            print(f"{'-'*60}")
            print(f"{'Total':<12} {total_neurons:<12} {overall_frequency:<12.6f}")
            print(f"{'='*60}")

        # Return the average membrane potential over time as the output
        # This is more stable for classification tasks
        output = torch.stack(mem_rec, dim=0).mean(dim=0)

        if debug:
            print(f"\nFinal output (avg membrane potential):")
            print(f"  shape: {output.shape}")
            print(f"  range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  mean: {output.mean():.4f}, std: {output.std():.4f}")
            print(f"{'='*60}\n")

        return output, overall_frequency

    def get_lif1_spike_frequency(self):
        """get lif1 layer spike frequency tensor"""
        return self.lif1_spike_frequency

    def get_lif1_frequency_stats(self):
        """get lif1 layer spike frequency statistics"""
        if self.lif1_spike_frequency is None:
            return None

        stats = {
            'mean_frequency': self.lif1_spike_frequency.mean().item(),
            'max_frequency': self.lif1_spike_frequency.max().item(),
            'min_frequency': self.lif1_spike_frequency.min().item(),
            'std_frequency': self.lif1_spike_frequency.std().item(),
            'frequency_per_neuron': self.lif1_spike_frequency.mean(dim=0)
        }
        return stats

    def get_all_spike_stats(self):
        """get all layers spike statistics"""
        if not hasattr(self, 'spike_stats'):
            return None
        return self.spike_stats

    def print_spike_stats(self):
        """print formatted spike statistics"""
        if not hasattr(self, 'spike_stats'):
            print("No spike statistics available. Run forward pass first.")
            return

        print(f"\n{'='*70}")
        print(f"Spike Firing Rate Statistics for All Layers")
        print(f"{'='*70}")
        print(f"{'Layer':<12} {'Neurons':<15} {'Frequency':<15} {'Contribution':<15}")
        print(f"{'-'*70}")

        total_neurons = self.spike_stats['total_neurons']
        for layer_name in ['lif1', 'layer1', 'layer2', 'layer3', 'layer4', 'lif_out']:
            stats = self.spike_stats[layer_name]
            neurons = stats['neurons']
            freq = stats['frequency']
            contribution = (neurons / total_neurons) * 100
            print(f"{layer_name:<12} {neurons:<15} {freq:<15.6f} {contribution:<15.2f}%")

        print(f"{'-'*70}")
        print(f"{'Total':<12} {total_neurons:<15} {self.spike_stats['overall_frequency']:<15.6f} {'100.00%':<15}")
        print(f"{'='*70}\n")


