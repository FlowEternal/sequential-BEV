import torch


def affine_grid_own(transformation, x):
    #  align_corners: if True, consider -1 and +1 to refer to the
    #  centers of the corner pixels rather than the image corners.
    batch_size, _, bev_feat_height, bev_feat_width = x.shape
    xs = (torch.arange(0, bev_feat_width, dtype=torch.float).view(1,bev_feat_width).expand(
        bev_feat_height, bev_feat_width) / (bev_feat_width -1) - 0.5) * 2
    ys = (torch.arange(0, bev_feat_height, dtype=torch.float).view(bev_feat_height,1).expand(
        bev_feat_height,bev_feat_width) / (bev_feat_height - 1) - 0.5 ) * 2
    ds = torch.tensor(1.0).view(1, 1).expand(bev_feat_height, bev_feat_width)
    frustum = torch.stack((xs, ys, ds), 0).view(3, -1).unsqueeze(0).repeat(batch_size,1,1).to(x.device)
    grid_own = torch.bmm( transformation, frustum )
    grid_own = grid_own.view(batch_size,2,bev_feat_height,bev_feat_width).permute(0,2,3,1).contiguous()
    return grid_own

if __name__ == '__main__':
    bev_feat_width = 64
    bev_feat_height = 32

    # input data
    transformation = torch.tensor([[[ 0.5362, -0.8441,  0.0690],[ 0.8441,  0.5362, -0.0039]]], device='cuda:0')
    print(transformation.shape)
    x = torch.randn([1,256,bev_feat_height,bev_feat_width], device="cuda:0")

    # pytorch implementation
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=True)
    print(grid.shape)

    # own implementation
    grid_own = affine_grid_own(transformation, x)
    print(grid_own.shape)

