Methodology
===========

Stage 1 — Starlet Wavelet Detection
------------------------------------

Each channel slice :math:`I_c \in \mathbb{R}^{H \times W}` is independently
analysed with the **undecimated isotropic wavelet transform** (à trous IUWT,
also known as the starlet transform).

The transform decomposes a 2-D image into :math:`J-1` detail bands plus a
coarse residual:

.. math::

   \mathcal{W}[I] = \{w_1, w_2, \ldots, w_{J-1}, c_{J-1}\}

where the detail coefficient at scale :math:`j` is

.. math::

   w_j = c_{j-1} - c_j, \qquad j = 1, \ldots, J-1

and each successive approximation :math:`c_j` is obtained by a separable
à trous B₃-spline convolution with dilation :math:`2^{j-1}`:

.. math::

   c_j = h^{(j)} \star c_{j-1}, \qquad
   h^{(j)}[k] = \tfrac{1}{16}\bigl[1,\,4,\,6,\,4,\,1\bigr]
   \text{ at step } 2^{j-1}

The B₃-spline kernel is applied in two separable passes (row then column)
via PyTorch dilated convolution, giving :math:`\mathcal{O}(5HW)` per scale
and :math:`\mathcal{O}(5JHW)` total, independent of :math:`j`.

Scale-adaptive thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The noise level at each detail scale is estimated from the
Median Absolute Deviation (MAD):

.. math::

   \hat{\sigma}_j = 1.4826 \times
   \operatorname{median}\bigl(|w_j - \operatorname{median}(w_j)|\bigr)

To prevent collapse of the per-channel MAD on nearly empty channels, the
noise reference is anchored to the **mean-map decomposition**:

.. math::

   \bar{I} = \frac{1}{|C|} \sum_{c \in C} I_c, \qquad
   \hat{\sigma}_j^{\text{ref}} = \hat{\sigma}_j(\bar{I}) \cdot \sqrt{|C|}

A pixel is declared significant if :math:`|w_j| > k_\sigma \cdot \hat{\sigma}_j^{\text{ref}}`,
where :math:`k_\sigma` is a user-controlled threshold (``k_sigma``).

Stage 2 — Masked TV-L1 Optical Flow
-------------------------------------

NEMO restricts the TV-L1 solver to the **union of source footprints** from
consecutive channel pair :math:`(c,\,c+1)`:

.. math::

   \Omega_{c,c+1} = \Bigl(\bigcup_b M_b^{(c)}\Bigr) \cup
                    \Bigl(\bigcup_b M_b^{(c+1)}\Bigr)

The flow field :math:`\mathbf{v}^* = (v_r, v_c) \in \mathbb{R}^{2 \times H \times W}`
is obtained by minimising:

.. math::

   \mathbf{v}^* = \underset{\mathbf{v}}{\arg\min}
   \bigl\|\nabla \tilde{I}_c + (\mathbf{v} \cdot \nabla)\tilde{I}_c\bigr\|_1
   + \lambda \|\nabla \mathbf{v}\|_1

The union mask (rather than intersection) is critical: when a source splits
into a new spatial location, the two components may not overlap, so an
intersection mask would yield :math:`\mathbf{v} = 0` and misclassify
the split-off component as a new independent source.

Stage 3 — Track Linking
------------------------

Track linking uses **advected masks** propagated channel-by-channel through
the TV-L1 flow via Catmull-Rom cubic interpolation.  Given a confirmed
component footprint :math:`M^{(t)}` and flow :math:`\mathbf{v}`, the
advected weight map is:

.. math::

   \mathcal{A}[M, \mathbf{v}](y', x') =
   \sum_{(y,x):\,M(y,x)=1}
   \delta\!\bigl(y' - \lfloor y + v_r(y,x)\rceil\bigr)\,
   \delta\!\bigl(x' - \lfloor x + v_c(y,x)\rceil\bigr)

Matching proceeds in four ordered steps per channel transition:

**A. Hungarian continuation** — negative pixel-overlap cost matrix
:math:`\mathcal{C}_{ij}` solved optimally; pairs above ``min_match_overlap``
are accepted as continuations.

**B. Centroid-distance fallback** — flow-extrapolated centroid matched to
unmatched components within ``max_gap_dist`` pixels (gap bridging).

**C. Merge detection** — tracks whose advected mask overlaps a component
already claimed by another track are classified as merges and deactivated.

**D. Split attribution** — unmatched components with advected-mask overlap
≥ ``min_split_overlap`` are attributed as splits of the best-matching parent.

Splits and merges are connected by union-find to form **sources** —
physical objects that may fragment and rejoin across channels.

Stage 4 — Kinematic Classification
------------------------------------

A track :math:`\tau` is **kinematically active** if its cumulative centroid
path length

.. math::

   \Delta_\tau = \sum_{k=1}^{N-1}
   \sqrt{(r_{k+1}-r_k)^2 + (x_{k+1}-x_k)^2} \;\geq\; \delta_{\min}

or if it was involved in any split or merge event.

Stage 5 — False-Detection Removal
------------------------------------

Each source is scored on two metrics:

**Flow-advection IoU** — measures how coherently the footprint follows the
flow field channel-to-channel.  Real sources score :math:`\text{IoU} \gtrsim 0.25`;
artefacts score near zero.

**Wavelet abruptness** — ratio of edge-channel wavelet flux to peak flux.
Step-function artefacts score :math:`\alpha \approx 1`; real sources fade
smoothly in and out.

A source is flagged as a false detection if:

.. math::

   \alpha > \alpha_{\text{thresh}} \quad\text{OR}\quad
   \bigl(\text{IoU}_{\text{flow}} < \text{IoU}_{\text{thresh}}
   \;\text{ AND }\; |C| < N_{\text{short}}\bigr)
