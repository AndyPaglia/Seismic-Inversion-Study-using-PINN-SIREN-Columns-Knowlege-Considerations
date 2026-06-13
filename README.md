# Knowledge-Constrained Full Waveform Inversion via SIREN/PINN:Incorporating Well Log Spatial Constraints into Neural Seismic Imaging

Full Waveform Inversion (FWI) is a high-resolution seismic imaging
technique that reconstructs subsurface velocity models by minimizing the differ-
ence between observed and synthetic seismic data.
Despite its effectiveness, FWI remains highly sensitive to the quality of the initial
model and dependent to converging to local minima in the absence of prior geo-
logical information.
This work proposes a knowledge-constrained extension of a SIREN/PINN-based
FWI framework, in which known velocity columns are incorporated into the inver-
sion process as hard spatial constraints.
Two constraint strategies are investigated: direct injection into the initial model
(Case A) and a soft penalization term added to the loss function (Case B).
The influence of three key parameters is systematically studied: the number of con-
strained columns, their spatial distribution (spaced vs. random), and the weighting
schedule of the constraint term (α).
Experiments conducted on three benchmark velocity models of increasing geolog-
ical complexity, Marmousi, Overthrust, and BP2004, demonstrate that Case B
consistently outperforms both the unconstrained baseline and Case A.
With 20 spaced columns and a linear α schedule, the proposed method achieves
a reduction in Mean Absolute Error of approximately 40% relative to the uncon-
strained baseline on the BP2004 model, while also accelerating convergence by a
factor of roughly two.
Uniform spatial distribution of the constraints is shown to be critical, as random
placement degrades both accuracy and convergence stability.
Among the α schedules tested, the linear ramp from 0.1 to 0.9 yields the best re-
sults across all models, with the performance gap over the sigmoid schedule being
largest on the most geologically complex benchmark.
These findings suggest that the incorporation of sparse but well-distributed well log
constraints into a physics-informed neural network framework represents a promis-
ing direction for improving the accuracy and robustness of seismic inversion in
complex geological settings.
