# Release verification

Verified on 10 November 2026 using generated CPU tensors only. No research data were read and no optimiser updates or additional research experiments were run.

- Five spectral backbones returned finite 64 x 64 predictions.
- Zero-initialised fusion exactly reproduced the backbone output.
- Bypassing the geological branch exactly reproduced the backbone output.
- Invalid geological pixels received zero injected residual.
- Maximum measured relative RMS was 0.010000001639127731, within float32 tolerance of the 0.01 bound.
- Removing the gate produced a unit gate; removing the cap retained the raw residual.
- Zero, very small, normal and large reference amplitudes produced finite bounded residuals and gradients.
- E3 and E4 metric implementations agreed on tied-score synthetic input and matched scikit-learn average precision.
- The supervised loss and gradients were finite, without an optimiser update.
- Scientific function bodies in both public engines matched the retained source at abstract-syntax-tree level. Only imports, local path binding and the opt-in command-line wrapper were adapted.

These checks validate the tested numerical properties, not the complete private input-preparation pipeline, the full CUDA training cycle or reproduction of reported paper metrics. Run `python examples/synthetic_smoke.py` to repeat the CPU checks in your environment.

## Documentation revision on 13 November 2026

The README, input contract and detailed usage guide were checked against the released CLI, data adapters, checkpoint format and inference function. Scientific Python files and dependency settings are unchanged from the verified release. This revision does not claim another full CUDA training run or access to the restricted inputs. The original numerical-check date above is retained.
