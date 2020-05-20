def test_net(net, loader):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    for _, it in enumerate(loader):
        batch_data = it[0]
        im1, im2, im3 = batch_data["im1"], batch_data["im2"], batch_data["im3"]
        outputs = net(im1, im2, im3)
    
    return im2, outputs