from evogym.viewer import EvoViewer
import time
from typing import Optional, Tuple, List
import numpy as np
from evogym.sim import EvoSim
from evogym.envs import EvoGymBase


class DingEvoViewer(EvoViewer):
    def __init__(self,
                 sim_to_view: EvoSim,
                 target_rps: Optional[int] = 50,
                 pos: Tuple[float, float] = (12, 4),
                 view_size: Tuple[float, float] = (40, 20),
                 resolution: Tuple[int, int] = (1200, 600)) -> None:

        super().__init__(sim_to_view, target_rps, pos, view_size, resolution)

    def render(self,
               mode: str = 'rgb_array',
               verbose: bool = False,
               hide_background: bool = False,
               hide_grid: bool = False,
               hide_edges: bool = False,
               hide_voxels: bool = False) -> Optional[np.ndarray]:
        """
        Render the simulation.
        Args:
            mode (str): values of 'screen' and 'human' will render to a debug window. If set to 'img' or 'rbg_array' will return an image array.
            verbose (bool): whether or not to print the rendering speed (rps) every second.
            hide_background (bool): whether or not to render the cream-colored background. If shut off background will be white.
            hide_grid (bool): whether or not to render the grid.
            hide_edges (bool): whether or not to render edges around all objects.
            hide_voxels (bool): whether or not to render voxels.
        Returns:
            Optional[np.ndarray]: if `mode` is set to `img`, will return an image array.
        """
        accepted_modes = ['screen', 'human', 'img', 'rgb_array']
        if not mode in accepted_modes:
            raise ValueError(
                f'mode {mode} is not a valid mode. The valid modes are {accepted_modes}'
            )

        self._init_viewer()
        render_settings = (hide_background, hide_grid, hide_edges, hide_voxels)

        current_time = self._sim.get_time()
        if current_time < self._last_rendered:
            self._last_rendered = current_time-50
        while self._last_rendered < current_time:
            self._update_tracking()
            self._last_rendered += 1

        out = None

        if mode == 'screen' or mode == 'human':
            if not self._is_showing_debug_window:
                self.show_debug_window()
                self._is_showing_debug_window = True
            if not self._has_init_screen_camera:
                self._init_screen_camera()
                self._has_init_screen_camera = True
            self._viewer.render(self.screen_camera, *render_settings)

        if mode in ['img', 'rgb_array']:
            if not self._has_init_img_camera:
                self._init_img_camera()
                self._has_init_img_camera = True
            self._viewer.render(self.img_camera, *render_settings)

            img_out = self.img_camera.get_image()
            img_out = np.array(img_out)
            img_out.resize(self.img_camera.get_resolution_height(),
                           self.img_camera.get_resolution_width(), 3)

            out = img_out

        self._timer.step(verbose=verbose)
        while not self._timer.should_step():
            time.sleep(0.001)

        return out
