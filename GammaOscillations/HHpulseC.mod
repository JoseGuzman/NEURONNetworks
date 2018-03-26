TITLE Action potential INa-IK with pulse-based kinetics

COMMENT
-----------------------------------------------------------------------------

	Pulse-based model for action potentials
	=======================================

MECHANISM

  Pulse-based formalism applied to sodium and potassium currents
  underlying action potentials.  The voltage-dependence of alpha's and beta's
  is approximated by a two-level function.  Based on the sharp transitions
  of alpha and beta during an action potential in the Hodgkin-Huxley model,
  these rate constants are assigned to a pulse when the membrane crosses
  a given threshold value (Vtr).   This allows to solve analytically all eqs
  for the variables m,n,h , leading to an extremely fast algorithm to generate
  action potentials (since no differential equation must be solved).


PARAMETERS

  Vtr: (mV) threshold of membrane potential for spike generation
  Dur: (ms) duration of the pulse
  Ref: (ms) refractory period after an action potential
  alphaM, betaM: (ms-1) max values of alpha and beta (variable m) 
  alphaH, betaH: (ms-1) max values of alpha and beta (variable h) 
  alphaN, betaN: (ms-1) max values of alpha and beta (variable n)
  gnabar, gkbar: (S/cm2) max conductances for INa and IK
  refract: (ms) absolute refractory period imposed, must be >dt

ASSIGNED VARIABLES

  m,n,h : activation and inactivation variables as in the Hodgkin-Huxley model.
             Their value is calculated using the analytical solution.
  spike : logical variable (1=spiking, 0=no spiking)
  lastspike : (ms) time at which the last spike occurred
  counter: used to define the pulse


WARNING

  This optimized version uses pre-calculated exponentials during initiaization.
  The time step CANNOT be changed during the simulation, otherwise dramatic
  integration errors will occur...
  To remove this optimization, use the code in the COMMENT block at the end 
  of this file.

For more information, please consult the web site: http://www.iaf.cnrs-gif.fr

  Alain Destexhe, 1995

  Modifs:
	correct pulse length, fast algorithm (update on m^3 and n^4)
	pulse implemented using a counter (more stable numerically)

-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX hhPC
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
	RANGE gnabar, gkbar
	RANGE Vtr, Dur, Ref, alphaM, betaM, alphaH, betaH, alphaN, betaN
	RANGE spike, expm1, expm2, exph1, exph2, expn1, expn2, counter
	RANGE lastspike, nocurrent, refract
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	gnabar	= .003 	(mho/cm2)	: max sodium conductance
	gkbar	= .005 	(mho/cm2)	: max potassium conductance
	ena	= 50	(mV)		: sodium reversal potential
	ek	= -90	(mV)		: potassium reversal potential
	dt              (ms)
	v              	(mV)

	Vtr 	= 0 	(mV)		: voltage threshold for spike
					: (heuristic: Vtr = vtraub+9.7)
	Dur	= 1	(ms)		: duration of the spike
	Ref 	= 1.5	(ms)		: mimimum time between spikes

	alphaM	= 10	(/ms)		: maximum value of fwd rate cst for m
	betaM	= 10	(/ms)		: maximum value of bkwd rate cst for m
	alphaH	= 0.3	(/ms)		: maximum value of fwd rate cst for h
	betaH	= 4	(/ms)		: maximum value of bkwd rate cst for h
	alphaN	= 2	(/ms)		: maximum value of fwd rate cst for n
	betaN	= 0.7	(/ms)		: maximum value of bkwd rate cst for n

        refract = 1     (ms)            : absolute refractory period (ms)
					: (must be at least equal to dt)
}


ASSIGNED {
	m				: activation of INa
	h				: inactivation of INa
	n				: activation of IK
	m3				: m^3
	n4				: n^4
	ina		(mA/cm2)	: Na+ current
	ik		(mA/cm2)	: K+ current
	spike				: spike flag
        lastspike       (ms)            : time of last spike
	counter		(ms)		: time counter
	nocurrent			: flag for integration
	expm1				: variables to store exponentials
	expm2
	exph1
	exph2
	expn1
	expn2
}

INITIAL {
	spike = 0
	m = 0
	h = 1
	n = 0
	m3 = 0
	n4 = 0
	expm1 = exp(-3*betaM*dt)	: power 3
	expm2 = exp(-alphaM*dt)
	exph1 = exp(-alphaH*dt)
	exph2 = exp(-betaH*dt)
	expn1 = exp(-4*betaN*dt)	: power 4
	expn2 = exp(-alphaN*dt)
	lastspike = -9e9
        nocurrent = 0
	counter = -10
}

BREAKPOINT {
	SOLVE fire
	if(nocurrent) {
		ina = 0
		ik = 0
	} else {
		ina = gnabar * m3*h * (v - ena)
		ik  = gkbar * n4 * (v - ek)
	}
}


PROCEDURE fire() {

	counter = counter - dt			: counter = time since last spike ended

						: ready for another spike?
	if (counter < -refract) {
		if (v > Vtr) {			: spike occured?
			spike = 1		: if yes, start new spike
			lastspike = t
			nocurrent = 0
			m = m3^0.33333333333333
                        n = n4^0.25
			counter = Dur		: set counter to pulse duration
		}
						
	} else if (counter > 0) {		: still spiking?
	
		: do nothing
	
	} else if (spike == 1) {		: terminate spike
			spike = 0
	}



	if (spike > 0) {				: spiking?
		m = 1 + (m-1) * expm2
		h = h * exph2
		n = 1 + (n-1) * expn2
		m3 = m*m*m
		n4 = n*n*n*n
	} else if (!nocurrent) {
		m3 = m3 * expm1
		h = 1 + (h-1) * exph1
		n4 = n4 * expn1
		if( (m<1e-9) && (n<1e-9) && ((1-h)<1e-9) ) {
			nocurrent = 1
			m = 0
			h = 1
			n = 0
			m3 = 0
			n4 = 0
		}
	}

}

COMMENT
: to be used to allow dt to be changed in the middle of the simulation

	if (spike > 0) {				: spiking?
		m = 1 + (m-1) * exp(-alphaM*dt)
		h = h * exp(-betaH*dt)
		n = 1 + (n-1) * exp(-alphaN*dt)
	} else {					: no spike occuring
		m = m * exp(-betaM*dt)
		h = 1 + (h-1) * exp(-alphaH*dt)
		n = n * exp(-betaN*dt)
	}
ENDCOMMENT
