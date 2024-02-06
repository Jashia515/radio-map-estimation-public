import torch
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = None#Encoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = None#Decoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def fit(self, train_dl, optimizer, epochs=1, loss='mse'):
        lt,lt1,ls=[],[],[]
        for epoch in range(epochs):
            lt1=lt
            lt=[]
            running_loss = 0.0
            
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                #print(t_x_point.shape)
                
                if epoch!=0:
                    t_x_point=lt1[i] # using the previous output of the model
                    
                    
                t_y_point_pred= self.forward(t_x_point).to(torch.float64)
                #print(t_y_point_pred.shape)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] losssssss: {running_loss/(i+1)}')
                
                #saving the current output
                rshp_y=t_y_point_pred.view(t_x_point.shape[0],1,32,32) # reshaping the 256 X 1024 vector to 256 X 1 X 32 X 32
                rshp_m=t_y_mask.view(t_x_point.shape[0],1,32,32) # reshaping the mask 256 X 1024 vector to 256 X 1 X 32 X 32
                aug_output=torch.cat([rshp_y,rshp_m],dim=1) # concatenating the outputs
                aug_output=aug_output.to(torch.float32).detach()
                
                lt.append(aug_output) # saving for next iteration
                ls.append(running_loss/(i+1))

        return running_loss/(i+1),ls


    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        
    def fit_wandb(self, train_dl, test_dl, scaler, optimizer, project_name, run_name, epochs=100, loss='mse'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            train_loss = self.fit(train_dl, optimizer, epochs=1, loss=loss)
            test_loss = self.evaluate(test_dl, scaler)
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

    def save_model(self, out_path):
        torch.save(self, out_path)