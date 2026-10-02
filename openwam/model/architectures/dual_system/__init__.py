"""DualSystem architecture family.

Four variants split into independent classes:

- :class:`DualSystemCrossAttnArchitecture` — bridge-collection plan, ActionDiT
  runs after the video DiT.
- :class:`DualSystemSelfAttnArchitecture` — interleaved plan, joint ActionDiT
  blocks run inside the video DiT loop.
- :class:`DualSystemIDMArchitecture` — two-stage IDM plan, video denoised first,
  then action denoised with frozen video as condition.
- :class:`DualSystemDoTArchitecture` — DoT plan (Faster-WAM), a shallow action
  head reads K/V fused from the video layers' clean prefix.
"""

from openwam.model.architectures.dual_system.dot import DualSystemDoTArchitecture
from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

__all__ = [
    "DualSystemCrossAttnArchitecture",
    "DualSystemSelfAttnArchitecture",
    "DualSystemIDMArchitecture",
    "DualSystemDoTArchitecture",
]
