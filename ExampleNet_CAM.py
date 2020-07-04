from ExampleNet import ExampleNet
from CAM_categorical import CAM


class ExampleNetCAM(ExampleNet):
    def __init__(
            self,
            out_channels,
    ):
        super(ExampleNetCAM, self).__init__(
            out_channels=out_channels,
        )

        self.cam = CAM(
            nclasses=out_channels,
        )

    def forward(
            self,
            img,
            atlas,
            atlas_label,
    ):
        """
        img [B 1 W+padding H+padding D+padding]: target scan intensity map after standardization.
        atlas [B 1 W H D]: atlas scan intensity map after standardization.
        atlas_label [B W H D C]: probabilistic atlas label
        """
        x = img
        for i, layer in enumerate(self.layers):
            x = layer(x)

        output_shape = x.shape
        edge_length = [(img.shape[2] - output_shape[2]) // 2,
                       (img.shape[3] - output_shape[3]) // 2,
                       (img.shape[4] - output_shape[4]) // 2]
        predictions = self.cam.forward(
            unary=x,
            img=img[:, :, edge_length[0]:-edge_length[0], edge_length[1]:-edge_length[1], edge_length[2]:-edge_length[2]],
            atlas=atlas,
            atlas_label=atlas_label,
        )

        return predictions


