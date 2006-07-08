/*
         Copyright (C) 2005 Matthew Davies and Paul Brossier

         This program is free software; you can redistribute it and/or modify
         it under the terms of the GNU General Public License as published by
         the Free Software Foundation; either version 2 of the License, or
         (at your option) any later version.

         This program is distributed in the hope that it will be useful,
         but WITHOUT ANY WARRANTY; without even the implied warranty of
         MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
         GNU General Public License for more details.

         You should have received a copy of the GNU General Public License
         along with this program; if not, write to the Free Software
         Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
         
*/

#include "aubio_priv.h"
#include "sample.h"
#include "mathutils.h"
#include "beattracking.h"

// 60*samplerate/winlen

/* maximum length for rp */
static smpl_t constthresh = 3.901; //empirically derived!

uint_t fvec_gettimesig(smpl_t * acf, uint_t acflen, uint_t gp);
void aubio_beattracking_checkstate(aubio_beattracking_t * bt);
smpl_t fvec_getperiod(aubio_beattracking_t * bt, uint_t timesig, uint_t rp);

/* could move to struct */
uint_t gp         = 0, bp  = 0, rp1 = 0, rp2 = 0, bp2 = 0;
smpl_t g_mu       = 0.;
smpl_t g_var      = 3.901;
uint_t flagconst  = 0;
uint_t flagstep   = 0;
// needs to be a signed ?
sint_t counter    = 0;
uint_t maxindex   = 0;
uint_t timesig    = 0;
uint_t rp         = 1;
uint_t lastbeat   = 0;
//number of harmonics in shift invariant comb filterbank
uint_t numelem    = 4;
smpl_t myperiod   = 0.;

uint_t maxnumelem = 4;

struct _aubio_beattracking_t {
        fvec_t * rwv;    /* rayleigh weight vector - rayleigh distribution function */                    
        fvec_t * gwv;    /* rayleigh weight vector - rayleigh distribution function */                    
        fvec_t * dfwv;   /* detection function weighting - exponential curve */
        fvec_t * dfrev;  /* reversed onset detection function */
        fvec_t * acf;    /* vector for autocorrelation function (of current detection function frame) */
        fvec_t * acfout; /* store result of passing acf through s.i.c.f.b. */
        fvec_t * phwv;   /* beat expectation alignment weighting */
        fvec_t * phout;
        //uint_t timesig;  /* time signature of input, set to zero until context dependent model activated */
        uint_t step;
        fvec_t * locacf; /* vector to store harmonics of filterbank of acf */
        fvec_t * inds;      /* vector for max index outputs for each harmonic */
        uint_t rayparam; /* Rayleigh parameter */
};

aubio_beattracking_t * new_aubio_beattracking(uint_t winlen,
                uint_t channels) {

        aubio_beattracking_t * p = AUBIO_NEW(aubio_beattracking_t);
        uint_t i        = 0;
	/* parameter for rayleigh weight vector - sets preferred tempo to
	 * 120bpm [43] */
	smpl_t rayparam = 48./512. * winlen;
        smpl_t dfwvnorm = EXP((LOG(2.0)/rayparam)*(winlen+2));
 	/** length over which beat period is found [128] */
        uint_t laglen   = winlen/4;
	/** step increment - both in detection function samples -i.e. 11.6ms or
	 * 1 onset frame [128] */
	uint_t step     = winlen/4; /* 1.5 seconds */

        p->rayparam = rayparam;
        p->step    = step;
        p->rwv     = new_fvec(laglen,channels);
        p->gwv     = new_fvec(laglen,channels);
        p->dfwv    = new_fvec(winlen,channels);
        p->dfrev   = new_fvec(winlen,channels);
        p->acf     = new_fvec(winlen,channels);
        p->acfout  = new_fvec(laglen,channels);
        p->phwv    = new_fvec(2*laglen,channels);
        p->phout   = new_fvec(winlen,channels);

        //p->timesig = 0;

        p->inds    = new_fvec(maxnumelem,channels);
        p->locacf  = new_fvec(winlen,channels); 

        /* exponential weighting, dfwv = 0.5 when i =  43 */
        for (i=0;i<winlen;i++) {
                p->dfwv->data[0][i] = (EXP((LOG(2.0)/rayparam)*(i+1)))
                        / dfwvnorm;
        } 

        for (i=0;i<(laglen);i++){
                p->rwv->data[0][i] = ((smpl_t)(i+1.) / SQR((smpl_t)rayparam)) * 
                        EXP((-SQR((smpl_t)(i+1.)) / (2.*SQR((smpl_t)rayparam))));
        }

        return p;

}

void del_aubio_beattracking(aubio_beattracking_t * p) {
        del_fvec(p->rwv);
        del_fvec(p->gwv);
        del_fvec(p->dfwv);
        del_fvec(p->dfrev);
        del_fvec(p->acf);
        del_fvec(p->acfout);
        del_fvec(p->phwv);
        del_fvec(p->phout);
        del_fvec(p->locacf);
        del_fvec(p->inds);
        AUBIO_FREE(p);
}


void aubio_beattracking_do(aubio_beattracking_t * bt, fvec_t * dfframe, fvec_t * output) {

        uint_t i,k;
        /* current beat period value found using gaussian weighting (from context dependent model) */
        uint_t step     = bt->step;
        uint_t laglen   = bt->rwv->length;
        uint_t winlen   = bt->dfwv->length;
        smpl_t * phout  = bt->phout->data[0];
        smpl_t * phwv   = bt->phwv->data[0];
        smpl_t * dfrev  = bt->dfrev->data[0];
        smpl_t * dfwv   = bt->dfwv->data[0];
        smpl_t * rwv    = bt->rwv->data[0];
        smpl_t * acfout = bt->acfout->data[0];
        smpl_t * acf    = bt->acf->data[0];
        //smpl_t * out    = output->data[0];

        //parameters for making s.i.c.f.b.
        uint_t a,b; 
        //beat alignment
        uint_t phase; 
        uint_t kmax;
        sint_t beat; 

        for (i = 0; i < winlen; i++){
                dfrev[winlen-1-i] = 0.;
                dfrev[winlen-1-i] = dfframe->data[0][i]*dfwv[i];
        }

        /* find autocorrelation function */
        aubio_autocorr(dfframe,bt->acf); 
        /*
        for (i = 0; i < winlen; i++){
                AUBIO_DBG("%f,",acf[i]);
        }
        AUBIO_DBG("\n");
        */

        /* get acfout - assume Rayleigh weightvector only */
        /* if timesig is unknown, use metrically unbiased version of filterbank */
        if(!timesig)  
                numelem = 4;
        //        AUBIO_DBG("using unbiased filterbank, timesig: %d\n", timesig);
        else
                numelem = timesig;
        //        AUBIO_DBG("using biased filterbank, timesig: %d\n", timesig);

        /* first and last output values are left intentionally as zero */
        for (i=0; i < bt->acfout->length; i++)
                acfout[i] = 0.;

        for(i=1;i<laglen-1;i++){ 
                for (a=1; a<=numelem; a++){
                        for(b=(1-a); b<a; b++){
                                acfout[i] += acf[a*(i+1)+b-1] 
                                        * 1./(2.*a-1.)*rwv[i];
                        }
                }
        }

        /* find non-zero Rayleigh period */
        maxindex = vec_max_elem(bt->acfout);
        rp = maxindex ? maxindex : 1;
        //rp = (maxindex==127) ? 43 : maxindex; //rayparam
        rp = (maxindex==bt->acfout->length-1) ? bt->rayparam : maxindex; //rayparam

        // get float period
        myperiod = fvec_getperiod(bt,timesig,rp);
        //AUBIO_DBG("\nrp =  %d myperiod = %f\n",rp,myperiod);
        //AUBIO_DBG("accurate tempo is %f bpm\n",5168./myperiod);

        /* activate biased filterbank */
        aubio_beattracking_checkstate(bt);
        /* end of biased filterbank */

        /* initialize output */
        for(i=0;i<bt->phout->length;i++)     {phout[i] = 0.;} 

        /* deliberate integer operation, could be set to 3 max eventually */
        kmax = winlen/bp;

        for(i=0;i<bp;i++){
                phout[i] = 0.;
                for(k=0;k<kmax;k++){
                        phout[i] += dfrev[i+bp*k] * phwv[i];
                }
        }

        /* find Rayleigh period */
        maxindex = vec_max_elem(bt->phout);
        if (maxindex == winlen-1) maxindex = 0;
        phase =  1 + maxindex;

        /* debug */
        //AUBIO_DBG("beat period = %d, rp1 = %d, rp2 = %d\n", bp, rp1, rp2);
        //AUBIO_DBG("rp = %d, gp = %d, phase = %d\n", rp, gp, phase);

        /* reset output */
        for (i = 0; i < laglen; i++)
                output->data[0][i] = 0.;

        i = 1;
        beat =  bp - phase;
        /* start counting the beats */
        if(beat >= 0)
        {
                output->data[0][i] = (smpl_t)beat;
                i++;
        }

        while( beat+bp < step )
        {
                beat += bp;
                output->data[0][i] = (smpl_t)beat;
                i++;
        }

        lastbeat = beat;
        /* store the number of beat found in this frame as the first element */
        output->data[0][0] = i;
}

uint_t fvec_gettimesig(smpl_t * acf, uint_t acflen, uint_t gp){
        sint_t k = 0;
        smpl_t three_energy = 0., four_energy = 0.;
        if( acflen > 6 * gp + 2 ){
                for(k=-2;k<2;k++){
                        three_energy += acf[3*gp+k];
                        four_energy += acf[4*gp+k];
                }
        }
        else{ /*Expanded to be more accurate in time sig estimation*/
                for(k=-2;k<2;k++){
                        three_energy += acf[3*gp+k]+acf[6*gp+k];
                        four_energy += acf[4*gp+k]+acf[2*gp+k];
                }
        }
        return (three_energy > four_energy) ? 3 : 4;
}

smpl_t fvec_getperiod(aubio_beattracking_t * bt, uint_t timesig, uint_t rp){
      	/*function to make a more accurate beat period measurement.*/

	smpl_t period = 0.;
	smpl_t maxval = 0.;
	
	sint_t a,b;
	uint_t i,j;	
	uint_t acfmi = rp; //acfout max index
	uint_t maxind = 0;

	if(!timesig)
		numelem = 4;
	else
		numelem = timesig;

	for (i=0;i<numelem;i++) // initialize
	bt->inds->data[0][i] = 0.;

	for (i=0;i<bt->locacf->length;i++) // initialize
                bt->locacf->data[0][i] = 0.;
	
	// get appropriate acf elements from acf and store in locacf
        for (a=1;a<=4;a++){
                for(b=(1-a);b<a;b++){		
                        bt->locacf->data[0][a*(acfmi)+b-1] = 
                                bt->acf->data[0][a*(acfmi)+b-1];		               	           	      
                }
        }

	for(i=0;i<numelem;i++){

		maxindex = 0;
		maxval = 0.0;
	
		for (j=0;j<(acfmi*(i+1)+(i)); j++){
                        if(bt->locacf->data[0][j]>maxval){
                                maxval = bt->locacf->data[0][j];
                                maxind = j;
                        }
                        //bt->locacf->data[0][maxind] = 0.;
                        bt->locacf->data[0][j] = 0.;
		}
		//AUBIO_DBG("\n maxind is  %d\n",maxind);
		bt->inds->data[0][i] = maxind;

	}

	for (i=0;i<numelem;i++){
		period += bt->inds->data[0][i]/(i+1.);}

	period = period/numelem;

	return (period);
}


void aubio_beattracking_checkstate(aubio_beattracking_t * bt) {
        uint_t i,j,a,b;
        uint_t laglen   = bt->rwv->length;
        uint_t acflen   = bt->acf->length;
        uint_t step     = bt->step;
        smpl_t * acf    = bt->acf->data[0];
        smpl_t * acfout = bt->acfout->data[0];
        smpl_t * gwv    = bt->gwv->data[0];
        smpl_t * phwv   = bt->phwv->data[0];

        if (gp) {
                // doshiftfbank again only if context dependent model is in operation
                //acfout = doshiftfbank(acf,gwv,timesig,laglen,acfout); 
                //don't need acfout now, so can reuse vector
                // gwv is, in first loop, definitely all zeros, but will have
                // proper values when context dependent model is activated
                for (i=0; i < bt->acfout->length; i++)
                       acfout[i] = 0.;
                for(i=1;i<laglen-1;i++){ 
                        for (a=1;a<=timesig;a++){
                                for(b=(1-a);b<a;b++){
                                        acfout[i] += acf[a*(i+1)+b-1] 
                                                * 1. * gwv[i];
                                }
                        }
                }
                gp = vec_max_elem(bt->acfout);
                /*
	        while(gp<32) gp =gp*2;
	        while(gp>64) gp = gp/2;
                */
        } else {
                //still only using general model
                gp = 0;  
        }

        //now look for step change - i.e. a difference between gp and rp that 
        // is greater than 2*constthresh - always true in first case, since gp = 0
        if(counter == 0){
                if(ABS(gp - rp) > 2.*constthresh) {
                        flagstep = 1; // have observed  step change.
                        counter  = 3; // setup 3 frame counter
                } else {
                        flagstep = 0;
                }
        }

        //i.e. 3rd frame after flagstep initially set
        if (counter==1 && flagstep==1) {
                //check for consistency between previous beatperiod values
                if(ABS(2.*rp - rp1 -rp2) < constthresh) {
                        //if true, can activate context dependent model
                        flagconst = 1;
                        counter   = 0; // reset counter and flagstep
                } else {
                        //if not consistent, then don't flag consistency!
                        flagconst = 0;
                        counter   = 2; // let it look next time
                }
        } else if (counter > 0) {
                //if counter doesn't = 1, 
                counter = counter-1;
        }

        rp2 = rp1; rp1 = rp; 

        if (flagconst) {
                /* first run of new hypothesis */
                gp = rp;
                g_mu = gp;
                timesig = fvec_gettimesig(acf,acflen, gp);
                for(j=0;j<laglen;j++)
                        gwv[j] = EXP(-.5*SQR((smpl_t)(j+1.-g_mu))/SQR(g_var));
                flagconst = 0;
                bp = gp;
                /* flat phase weighting */
                for(j=0;j<2*laglen;j++)  {phwv[j] = 1.;} 
        } else if (timesig) {
                /* context dependant model */
                bp = gp;
                /* gaussian phase weighting */
                if (step > lastbeat) {
                        for(j=0;j<2*laglen;j++)  {
                                phwv[j] = EXP(-.5*SQR((smpl_t)(1.+j-step+lastbeat))/(bp/8.));
                        }
                } else { 
                        //AUBIO_DBG("NOT using phase weighting as step is %d and lastbeat %d \n",
                        //                step,lastbeat);
                        for(j=0;j<2*laglen;j++)  {phwv[j] = 1.;} 
                }
        } else {
                /* initial state */ 
                bp = rp;
                /* flat phase weighting */
                for(j=0;j<2*laglen;j++)  {phwv[j] = 1.;} 
        }

        /* do some further checks on the final bp value */

        /* if tempo is > 206 bpm, half it */
        while (bp < 25) {
                //AUBIO_DBG("warning, doubling the beat period from %d\n", bp);
                //AUBIO_DBG("warning, halving the tempo from %f\n", 60.*samplerate/hopsize/bp);
                bp = bp*2;
        }
        
        //AUBIO_DBG("tempo:\t%3.5f bpm | ", 5168./bp);

        /* smoothing */
        //bp = (uint_t) (0.8 * (smpl_t)bp + 0.2 * (smpl_t)bp2);
        //AUBIO_DBG("tempo:\t%3.5f bpm smoothed | bp2 %d | bp %d | ", 5168./bp, bp2, bp);
        //bp2 = bp;
        //AUBIO_DBG("time signature: %d \n", timesig);

}
