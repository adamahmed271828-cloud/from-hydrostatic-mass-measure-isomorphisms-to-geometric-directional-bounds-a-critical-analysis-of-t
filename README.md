This paper establishes a formal mathematical continuum bridging
classical hydrostatic mass-proportional transformations with the global regularity
and finite-time blow-up problems of the three-dimensional incompressible Navier–
Stokes equations.
We begin by constructing a Lebesgue-measure isomorphism from a static fluid
analogy, recovering geometric domain areas without appeal to transcendental
constants (Theorem 2.2). Extending this to time-dependent kinematics via the
Reynolds Transport Theorem, we prove that the global Lebesgue measure μ(Ω(t))
is an invariant of any incompressible flow (Theorem 4.1).
We then carry out a careful micro-scale localisation around a suspected singular
set S, applying a three-fold H¨older inequality to the local convective–dissipation
interaction (Proposition 5.2). A rigorous peer-review of the resulting estimates
reveals two explicit obstructions: (i) a circular-logic loop arising from the need to
control higher-order Sobolev norms, and (ii) a non-local pressure contribution that
cannot be dominated by purely local volume bounds due to infrared divergences in
the Riesz kernel.
We then reformulate the entire problem as the Viscous Dominance Conjecture:
the existence of a universal constant δ > 0 such that the vortex-stretching integral
never exceeds (1 − δ) times the viscous dissipation (Conjecture 7.1). If true, this
conjecture implies global enstrophy decay and, consequently, global regularity. We
identify the geometric origin of δ in the trace-free structure of the strain-rate tensor
and in the directional alignment constraints studied by Constantin and Fefferman
(7).
The paper includes a deterministic C++ Lagrangian simulation (RK4 integration,
N = 1000 particles) that confirms mass-measure conservation at machine precision
under chaotic non-linear advection (Section 11), together with visualisations of
the three-dimensional velocity field, vorticity tubes, pressure map, and Littlewood–
Paley Besov spectrum.
Disclaimer. The present paper does not claim to resolve the Clay Millennium
Problem. Every unproved statement is explicitly labelled as a Conjecture or Open
Problem. The paper is presented as a structured research programme whose
objective is to identify the precise remaining bottlenecks and to propose concrete
sub-problems whose resolution would advance the global theory.
