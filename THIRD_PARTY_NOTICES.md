# Third-party notices

## E-SAM

The expert-choice noisy routing logic in `c2vlm_model.py` is adapted from
[Asphyxiate-Rye/E-SAM](https://github.com/Asphyxiate-Rye/E-SAM), specifically
`model/MoE.py`. E-SAM is distributed under the MIT License; the license is
included in `third_party/E-SAM_LICENSE`.

The integration preserves E-SAM's expert-choice direction, noisy router,
four-times MLP experts, capacity-factor calculation, residual expert output,
and cross-stage attention. Device-hardcoded allocation and mixed-precision
dtype issues were corrected for this runnable C2-VLM integration.
