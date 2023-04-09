"""
Function: Grid Sample Cuda Implementation for ONNX Export
Author: Zhan Dong Xu
Date: 2021/11/11
"""

import torch

def bilinear_sample_noloop(image, grid):
    """
    :param image: sampling source of shape [N, C, H, W]
    :param grid: integer sampling pixel coordinates of shape [N, grid_H, grid_W, 2]
    :return: sampling result of shape [N, C, grid_H, grid_W]
    """
    Nt, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]

    grid[:,:,:,0] = (grid[:,:,:,0] * 0.5 + 0.5) * (W - 1)
    grid[:,:,:,1] = (grid[:,:,:,1] * 0.5 + 0.5) * (H - 1)

    xgrid, ygrid = grid.split([1, 1], dim=-1)
    mask = ((xgrid >= 0.0) & (ygrid >= 0.0) & (xgrid <= float(W - 1) ) & (ygrid <= float(H - 1) )).float()

    x0 = torch.floor(xgrid)
    x1 = x0 + 1
    y0 = torch.floor(ygrid)
    y1 = y0 + 1
    wa = ((x1 - xgrid) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wb = ((x1 - xgrid) * (ygrid - y0)).permute(3, 0, 1, 2)
    wc = ((xgrid - x0) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wd = ((xgrid - x0) * (ygrid - y0)).permute(3, 0, 1, 2)
    x0 = (x0 * mask).view(Nt, grid_H, grid_W).long()
    y0 = (y0 * mask).view(Nt, grid_H, grid_W).long()
    x1 = (x1 * mask).view(Nt, grid_H, grid_W).long()
    y1 = (y1 * mask).view(Nt, grid_H, grid_W).long()
    ind = torch.arange(0, Nt, device=image.device) #torch.linspace(0, Nt - 1, Nt, device=image.device)
    ind = ind.view(Nt, 1).expand(-1, grid_H).view(Nt, grid_H, 1).expand(-1, -1, grid_W).long()
    image = image.permute(1, 0, 2, 3)

    output_tensor = (image[:, ind, y0, x0] * wa +
                     image[:, ind, y1, x0] * wb +
                     image[:, ind, y0, x1] * wc +
                     image[:, ind, y1, x1] * wd).permute(1, 0, 2, 3)

    output_tensor *= mask.permute(0, 3, 1, 2).expand(-1, C, -1, -1)
    return output_tensor

if __name__ == '__main__':
    image = torch.randn([1, 128, 30, 40]).cuda()
    grid = torch.randn([1, 32, 64, 2]).cuda()
    output = bilinear_sample_noloop(image, grid)
    print("Grid Sample Output Shape: ")
    print(output.shape)

    import time
    for _ in range(10):
        tic = time.time()
        torch.cuda.synchronize()
        output = bilinear_sample_noloop(image, grid)
        torch.cuda.synchronize()
        print("inference time is %.2f " %( 1000*( time.time() - tic ) ) )