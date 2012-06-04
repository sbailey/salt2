"""
Utilities for working with SALT2 templates

Stephen Bailey
Winter 2007
"""

import numpy
import numpy.random
import os
from scipy.optimize.optimize import fmin
### from scipy.interpolate import interp1d
from glob import glob

def model(modeldir=None):
    """Create and return a Salt2Model class instance.  Same as Salt2Model(modeldir)."""
    return Salt2Model(modeldir)

class interp1d(object):
    """
    Utility class for providing functionality like
    scipy.interpolate.interp1d while also providing left/right fill values
    like numpy.interp.  Sigh.
    """
    def __init__(self, x, y):
        """
        Return a callable f such that y = f(x)
        """
        self.x = x
        self.y = y
        
    def __call__(self, x):
        return numpy.interp(x, self.x, self.y)

#-------------------------------------------------------------------------
class TimeSeries:
    """
    A series of values associated with a phase and a wavelength,
    e.g. a time series of spectra or a time series of their errors
    """
    
    def __init__(self, filename):
        """
        Initialize with ASCII file with grid data in the form
          phase wavelength value
        """
        self._wavelenghts = None  #- Wavelenghts of first day, assume others are the same
        self._phases = []         #- Phases in the model file
        self._spectra = []        #- One spectrum interpolation function for each phase
        
        currentday = None
        w = []
        flux = []
        for line in open(filename):
            day, wavelength, value = map(float, line.split())
            if currentday == None:
                currentday = day

            if day != currentday:
                self._phases.append(currentday)
                self._spectra.append(interp1d(w, flux))
                if self._wavelenghts is None:
                    self._wavelengths = numpy.array(w)
                currentday = day
                w = []
                flux = []
            
            w.append(wavelength)
            flux.append(value)
            
        #- Get the last day of information in there 
        self._phases.append(currentday)
        self._spectra.append(interp1d(w, flux))


    def spectrum(self, phase, wavelengths=None, extend=True):
        """
        Return spectrum at requested phase and wavelengths.
        Raise ValueError if phase is out of range of model unless extend=True.
        """
        #- Bounds check first
        if phase < self._phases[0] and not extend:
            raise ValueError, "phase %.2f before first model phase %.2f" % (phase, self._phases[0])
        if phase > self._phases[-1] and not extend:
            raise ValueError, "phase %.2f after last model phase %.2f" % (phase, self._phases[-1])

        #- Use default wavelengths if none are specified
        if wavelengths is None:
            wavelengths = self._wavelengths

        #- Check if requested phase is out of bounds or exactly in the list
        if phase in self._phases:
            iphase = self._phases.index(phase)
            return self._spectra[iphase](wavelengths)
        elif phase < self._phases[0]:
            return self._spectra[0](wavelengths)
        elif phase > self._phases[-1]:
            return self._spectra[-1](wavelengths)
            
        #- If we got this far, we need to interpolate phases
        i = numpy.searchsorted(self._phases, phase)
        speclate = self._spectra[i](wavelengths)
        specearly = self._spectra[i-1](wavelengths)
        dphase = (phase - self._phases[i-1]) / (self._phases[i] - self._phases[i-1] )
        dspec = speclate - specearly
        spec = specearly + dphase*dspec
        
        return spec

    def wavelengths(self):
        """Return array of wavelengths sampled in the model"""
        return self._wavelengths.copy()

    def phases(self):
        """Return array of phases sampled in the model"""
        return numpy.array(self._phases)
        
    def grid(self):
        """Return a 2D array of spectrum[phase, wavelength]"""
        nspec = len(self._phases)
        nwave = len(self._wavelengths)
        z = numpy.zeros((nspec, nwave), dtype='float64')
        for i, spec in enumerate(self._spectra):
            z[i, :] = spec(self._wavelengths)
        return z


class Salt2Model:
    """
    An interface class to the SALT2 supernovae spectral timeseries model.
    """

    def __init__(self, modeldir=None):
        """
        Create a SALT2 model, using the model files found under modeldir,
        which is equivalent to the PATHMODEL environment variable for SALT2.
        i.e. it expects to find files in modeldir/salt2*/salt2*.dat
        
        It will use $PATHMODEL/salt2 if it is there, otherwise it will
        use the latest version in $PATHMODEL/salt2* .
        """
        if modeldir is None:
            if 'PATHMODEL' not in os.environ:
                raise ValueError, "If you don't specify modeldir, you must have salt2 PATHMODEL env var defined"
                
            pathmodel = os.environ['PATHMODEL']
            if os.path.exists(pathmodel+'/salt2/'):
                modeldir =  pathmodel+'/salt2/'
            else:
                models = sorted(glob(pathmodel + '/salt2*'))
                if len(models) == 0:
                    raise ValueError, "Model directory %s/salt2* not found" % pathmodel
                modeldir = models[-1]
                
            print "Using model", modeldir

        self._model = dict()
        self._model['M0'] = TimeSeries(modeldir+'/salt2_template_0.dat')
        self._model['M1'] = TimeSeries(modeldir+'/salt2_template_1.dat')

        self._model['V00'] = TimeSeries(modeldir+'/salt2_spec_variance_0.dat')
        self._model['V11'] = TimeSeries(modeldir+'/salt2_spec_variance_1.dat')
        self._model['V01'] = TimeSeries(modeldir+'/salt2_spec_covariance_01.dat')
        errorscale = modeldir+'/salt2_spec_dispersion_scaling.dat'
        if os.path.exists(errorscale):
            self._model['errorscale'] = TimeSeries(errorscale)
        
        self._wavelengths = self._model['M0'].wavelengths()
        

    def wavelengths(self):
        """Return wavelength coverage array of the model"""
        return self._wavelengths.copy()

    def flux(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """Return the model (wavelength, flux) spectrum for these parameters"""
        if wavelengths is None:
            wavelengths = self._wavelengths
        f0 = self._model['M0'].spectrum(phase, wavelengths=wavelengths)
        f1 = self._model['M1'].spectrum(phase, wavelengths=wavelengths)
        flux = x0*(f0 + x1*f1)

        flux *= self._extinction(wavelengths, c)

        return flux

    def error(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """
        Return flux error spectrum for given parameters
        """
        if wavelengths is None:
            wavelengths = self._wavelengths
            
        v00 = self._model['V00'].spectrum(phase, wavelengths=wavelengths)
        v11 = self._model['V11'].spectrum(phase, wavelengths=wavelengths)
        v01 = self._model['V01'].spectrum(phase, wavelengths=wavelengths)

        sigma = x0 * numpy.sqrt(v00 + x1*x1*v11 + 2*x1*v01)
        sigma *= self._extinction(wavelengths, c)
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errorscale' in self._model:
            sigma *= self._model['errorscale'].spectrum(phase, wavelengths=wavelengths)
        
        #- Hack adjustment to error (from SALT2 code)
        if phase < -7 or phase > +12:
            xx = numpy.nonzero(wavelengths < 3400)
            sigma[xx] *= 1000
        
        return sigma

    def variance(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """
        Return model flux variance for given parameters.
        """
        return self.error(phase, wavelengths=wavelengths, x0=x0, x1=x1, c=c)**2

    def fitspectra(self, spectra, phase=None, x0=None, frac=1.0):
        """
        Fit a set of spectra for jdmax, x0, x1, c.
        'spectra' is a list, each element is another list containing
            [juliandate, wavelength, flux, fluxerr]
            wavelength, flux, fluxerr are lists of values for that spectrum
        Returns jdmax, x0, x1, c  (sorry, no errors)
        """
        
        #- Get x0 normalization into the right ballpark
        maxflux = 0.0
        jdmaxguess = 0.0
        for jd, w, flux, fluxerr in spectra:
            fluxsum = sum(flux)
            if fluxsum > maxflux:
                maxflux = fluxsum
                jdmaxguess = jd
        
        if x0 is None:
            modelsum = sum( self.flux(phase=0, wavelengths=w) )
            x0 = maxflux/modelsum
            print "x0 starting guess", x0

        if phase is None:
            jdmax = jdmaxguess
        else:
            jdmax = spectra[0][0] - phase

        p0 = (jdmax, x0, 0.1, 0.1)  #- jdmax, x0, x1, c
        pfit = fmin(self.chi2, p0, args=(spectra, frac))
        return pfit

    def chi2(self, params, spectra, frac):
        """
        Utility function for fitspectra minimizer.
        """
        datemax, x0, x1, c = params
        c2 = self.chi2spectra(spectra, jdmax=datemax, x0=x0, x1=x1, c=c, \
                        normchi2=True, frac=frac) 
        ### print "--> %f  %g  %g  %g : %g" % (datemax, x0, x1, c, c2)
        return c2

    def chi2spectra(self, spectra, jdmax=0.0, x0=1.0, x1=0.0, c=0.0, \
                    normchi2=False, ngood=False, frac=1.0):
        """
        Perform a simultaneous chi2 calculation of a set of spectra.
        'spectra' is a list, each element is another list containing
            [juliandate, wavelength, flux, fluxerr]
        'jdmax' is julian date of B-band maximum; phases are calculated
            relative to this date.
        Do not include points in fit with modelerr/model > 100.
        If normchi2=True, normalize chi2 by number of points used (~dof)
        If ngood=True, return (chi2, ngoodpoints) instead of just chi2
        """
        
        ngoodpoints = 0
        chi2 = 0.0
        for jd, w, flux, fluxerr in spectra:
            phase = jd - jdmax
            mflux = self.flux(phase=phase, wavelengths=w, x0=x0, x1=x1, c=c)
            mfluxerr = self.error(phase=phase, wavelengths=w, x0=x0, x1=x1, c=c)
            
            mfluxerr = 0.0  ### don't include model error 
            
            diff = flux - mflux
            err2 = fluxerr**2 + mfluxerr**2
            badmodel = numpy.nonzero(mfluxerr > 100*mflux)[0]
            diff[badmodel] = 0.0
            ngoodpoints += len(diff) - len(badmodel)
            xchi2 = diff**2 / err2
            if frac < 1.0:
                xchi2.sort()
                nfrac = len(xchi2)*frac
                xchi2 = xchi2[0:nfrac]
            chi2 += sum( xchi2 )

        if normchi2:
            chi2 /= ngoodpoints
            
        if ngood:
            return chi2, ngoodpoints
        else:
            return chi2

    def chi2spectrum(self, wavelength, flux, fluxerr,
                   phase=0.0, x0=1.0, x1=0.0, c=0.0, ngood=False,
                   frac=1.0):
        """
        Calc chi2 fit for a single spectrum.  Wrapper for chi2spectra().
        """
        spectrum = [phase, wavelength, flux, fluxerr]
        return self.chi2spectra( [spectrum,], jdmax=0.0, \
                                    x0=x0, x1=x1, c=c, ngood=ngood, frac=frac) 

    def _extinction(self, w, c=0.1, params=(-0.336646, +0.0484495)):
        """
        From SALT2 code comments:
          ext = exp( color * constant * ( l + params(0)*l^2 + params(1)*l^3 + ... ) / ( 1 + params(0) + params(1) + ... ) )
              = exp( color * constant *  numerator / denominator )
              = exp( color * expo_term ) 
        """
        from math import log
        from numpy import exp
        const = log(10)/2.5

        if type(w) != numpy.ndarray:
            w = numpy.array(w)

        wB = 4302.57
        wV = 5428.55
        wr = (w-wB) / (wV-wB)

        numerator = 1.0*wr
        denominator = 1.0

        wi = wr*wr
        for p in params:
            numerator += wi*p
            denominator += p
            wi *= wr

        return exp(c * const * numerator / denominator )

if __name__ == '__main__':
    import pylab
    m = model()
    w = m.wavelengths()
    f = m.flux(0)
    
    e = m.error(0)
    x = numpy.concatenate( (w, w[::-1]) )
    y = numpy.concatenate( (f-e, (f+e)[::-1]) )
    gray = (0.8, 0.8, 0.8)
    pylab.fill(x, y, facecolor=gray, edgecolor=gray)
    
    pylab.plot(w, f)
    ww = numpy.arange(5000, 6001, 2)
    pylab.plot(ww, m.flux(0, wavelengths=ww), 'r,')
    
    pylab.show()
    





