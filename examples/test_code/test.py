from pyshearlab import MirrorFilt, dfilters, modulate2, SLgetShearletIdxs2D, SLpadArray, SLupsample, SLdshear
import numpy as np
import scipy
import torch

fftlib = np.fft

def getPassFilts(
    rows,
    cols,
    shearLevels,
    scalingFilter=None,
):
    """
    Computes the wedge, bandpass and lowpass filter for 2D shearlets. If no
    directional filter, scaling filter and wavelet filter are given, some
    standard filters are used.

    rows, cols and shearLevels are mandatory.

    in this function as long as the wavelet filter is unsupplied (the expected interal usage) the wavelet filter will be the mirror of the scaling filter.
    """
    if scalingFilter is None:
        scalingFilter = np.array(
            [
                0.0104933261758410,
                -0.0263483047033631,
                -0.0517766952966370,
                0.276348304703363,
                0.582566738241592,
                0.276348304703363,
                -0.0517766952966369,
                -0.0263483047033631,
                0.0104933261758408,
            ]
        )

    waveletFilter = MirrorFilt(scalingFilter)

    # initialize variables

    # get number of scales
    NScales = 1

    # allocate bandpass and wedge filters
    bandpass = np.zeros(
        (rows, cols, NScales), dtype=complex
    )  # these filters partition the frequency plane into different scales

    wedge = [None] * (
        max(shearLevels) + 1
    )  # these filters partition the frequenecy plane into different directions

    filterHigh = [None] * NScales
    filterLow = [None] * NScales
    filterLow2 = [None] * (max(shearLevels) + 1)

    filterHigh[-1] = waveletFilter
    filterLow[-1] = scalingFilter  # this filter corresponds to h_1 on page 11
    filterLow2[-1] = scalingFilter


    for j in range(len(filterHigh) - 2, -1, -1):
        filterLow[j] = np.convolve(filterLow[-1], SLupsample(filterLow[j + 1], 2, 1))
        filterHigh[j] = np.convolve(filterLow[-1], SLupsample(filterHigh[j + 1], 2, 1))

    h0, _ = dfilters("dmaxflat4", "d") 
    h0 /= np.sqrt(2)
    directionalFilter = modulate2(h0, "c")

    # normalize filters
    directionalFilter = directionalFilter / sum(sum(np.absolute(directionalFilter)))

    # compute wavelet high- and lowpass filters associated with a 1D Digital
    # wavelet transform on Nscales scales, e.g., we compute h_1 to h_J and
    # g_1 to g_J (compare page 11) with J = nScales.
    for j in range(len(filterHigh) - 2, -1, -1):
        filterLow[j] = np.convolve(filterLow[-1], SLupsample(filterLow[j + 1], 2, 1))
        filterHigh[j] = np.convolve(filterLow[-1], SLupsample(filterHigh[j + 1], 2, 1))

    for j in range(len(filterLow2) - 2, -1, -1):
        filterLow2[j] = np.convolve(filterLow2[-1], SLupsample(filterLow2[j + 1], 2, 1))

    # construct bandpass filters for scales 1 to nScales
    for j in range(len(filterHigh)):
        bandpass[:, :, j] = fftlib.fftshift(
            fftlib.fft2(
                fftlib.ifftshift(SLpadArray(filterHigh[j], np.array([rows, cols])))
            )
        )
    bandpass = fftlib.fftshift(
        fftlib.fft2(
            fftlib.ifftshift(
                SLpadArray(filterHigh[0], np.array([rows, cols]))
            )
        )
    )
    highlowpass = fftlib.fftshift(
        fftlib.fft2(
            fftlib.ifftshift(
                SLpadArray(np.outer(filterHigh[0], filterLow[0]), np.array([rows, cols]))
            )
        )
    )
    lowhighpass = fftlib.fftshift(
        fftlib.fft2(
            fftlib.ifftshift(
                SLpadArray(np.outer(filterLow[0], filterHigh[0]), np.array([rows, cols]))
            )
        )
    )
    lowpass = fftlib.fftshift(
        fftlib.fft2(
            fftlib.ifftshift(
                SLpadArray(np.outer(filterLow[0], filterLow[0]), np.array([rows, cols]))
            )
        )
    )

    filterLow2[-1].shape = (1, len(filterLow2[-1]))
    for shearLevel in np.unique(shearLevels):

        wedge[shearLevel] = np.zeros(
            (rows, cols, int(np.floor(np.power(2, shearLevel + 1) + 1))), dtype=complex
        )

        directionalFilterUpsampled = SLupsample(
            directionalFilter, 1, np.power(2, shearLevel + 1) - 1
        )

        filterLow2[-1 - shearLevel].shape = (1, len(filterLow2[-1 - shearLevel]))

        wedgeHelp = scipy.signal.convolve2d(
            directionalFilterUpsampled,
            np.transpose(filterLow2[len(filterLow2) - shearLevel - 1]),
        )
        wedgeHelp = SLpadArray(wedgeHelp, np.array([rows, cols]))

        wedgeUpsampled = SLupsample(wedgeHelp, 2, np.power(2, shearLevel) - 1)

        lowpassHelp = SLpadArray(
            filterLow2[len(filterLow2) - max(shearLevel - 1, 0) - 1],
            np.asarray(wedgeUpsampled.shape),
        )
        if shearLevel >= 1:
            wedgeUpsampled = fftlib.fftshift(
                fftlib.ifft2(
                    fftlib.ifftshift(
                        fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(lowpassHelp)))
                        * fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(wedgeUpsampled)))
                    )
                )
            )
        lowpassHelpFlip = np.fliplr(lowpassHelp)
        # traverse all directions of the upper part of the left horizontal
        # frequency cone
        for k in range(-np.power(2, shearLevel), np.power(2, shearLevel) + 1):
            # resample wedgeUpsampled as given in equation (22) on page 15.
            wedgeUpsampledSheared = SLdshear(wedgeUpsampled, k, 2)
            # convolve again with flipped lowpass filter, as required by
            # equation (22) on page 15
            if shearLevel >= 1:
                wedgeUpsampledSheared = fftlib.fftshift(
                    fftlib.ifft2(
                        fftlib.ifftshift(
                            fftlib.fftshift(
                                fftlib.fft2(fftlib.ifftshift(lowpassHelpFlip))
                            )
                            * fftlib.fftshift(
                                fftlib.fft2(fftlib.ifftshift(wedgeUpsampledSheared))
                            )
                        )
                    )
                )
            # obtain downsampled and renormalized and sheared wedge filter
            # in the frequency domain, according to equation (22), page 15.
            wedge[shearLevel][:, :, int(np.fix(np.power(2, shearLevel)) - k)] = (
                fftlib.fftshift(
                    fftlib.fft2(
                        fftlib.ifftshift(
                            np.power(2, shearLevel)
                            * wedgeUpsampledSheared[
                                :,
                                0 : np.power(2, shearLevel) * cols - 1 : np.power(
                                    2, shearLevel
                                ),
                            ]
                        )
                    )
                )
            )

    return wedge, bandpass, highlowpass, lowhighpass, lowpass

def prepareFilters(
    rows,
    cols,
    shearLevels=None,
    directionalFilter=None,
    quadratureMirrorFilter=None,
):

    nScales=1,
    scalingFilter = quadratureMirrorFilter
    scalingFilter2, waveletFilter = None, None
    # check input arguments
    if shearLevels is None:
        shearLevels = np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
    if scalingFilter is None:
        scalingFilter = np.array(
            [
                0.0104933261758410,
                -0.0263483047033631,
                -0.0517766952966370,
                0.276348304703363,
                0.582566738241592,
                0.276348304703363,
                -0.0517766952966369,
                -0.0263483047033631,
                0.0104933261758408,
            ]
        )
    if scalingFilter2 is None:
        scalingFilter2 = scalingFilter
    if directionalFilter is None:
        h0, h1 = dfilters("dmaxflat4", "d")
        h0 /= np.sqrt(2)
        h1 /= np.sqrt(2)
        directionalFilter = modulate2(h0, "c")
    if waveletFilter is None:
        waveletFilter = MirrorFilt(scalingFilter)

    fSize = np.array([rows, cols])
    filters = {"size": fSize, "shearLevels": shearLevels}
    wedge, bandpass, highlowpass, lowhighpass, lowpass = getPassFilts(
        rows,
        cols,
        shearLevels,
        scalingFilter,
    )
    wedge[0] = 0  # for matlab compatibilty (saving filters as .mat files)
    filters["cone1"] = {"wedge": wedge, "bandpass": bandpass, "lowpass": lowpass, "highlowpass": highlowpass, "lowhighpass": lowhighpass}
    if rows == cols:
        filters["cone2"] = filters["cone1"]
    else:
        wedge2, bandpass, highlowpass, lowhighpass, lowpass = getPassFilts(
            cols,
            rows,
            shearLevels,
            scalingFilter,
        )
        wedge2[0] = 0  # for matlab compatibilty (saving filters as .mat files)
        filters["cone2"] = {"wedge": wedge2, "bandpass": bandpass, "lowpass": lowpass, "highlowpass": highlowpass, "lowhighpass": lowhighpass}
    return filters

def getShearlets(preparedFilters, shearletIdxs=None):
    shearletIdxs = SLgetShearletIdxs2D(preparedFilters["shearLevels"]) if shearletIdxs is None else shearletIdxs
    nShearlets = shearletIdxs.shape[0]

    bandpass_shearlets = []
    highlowpass_shearlets = []
    lowhighpass_shearlets = []
    lowpass_shearlets = []

    rows, cols = preparedFilters["size"][0], preparedFilters["size"][1]

    # compute shearlets
    for j in range(nShearlets):
        cone = shearletIdxs[j, 0]
        scale = shearletIdxs[j, 1]
        shearing = shearletIdxs[j, 2]
        if cone == 0:
            # here we sort of plan to do nothing, but if it really comes down to it we can use the lowpass filter here if we want.
            # shearlets[:, :, j] = preparedFilters["cone1"]["lowpass"]
            lowpass_shearlets.append(preparedFilters["cone1"]["lowpass"])
            pass
        elif cone == 1:
            """
            shearlets[:, :, j] = preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, -shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone1"]["bandpass"][:, :, scale - 1])
            """
            bandpass_shearlet = preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, -shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone1"]["bandpass"])

            bandpass_shearlets.append(bandpass_shearlet)         
            
            lowpass_shearlet = preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, -shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone1"]["lowpass"])

            lowpass_shearlets.append(lowpass_shearlet)            
            
            highlowpass_shearlet = preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, -shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone1"]["highlowpass"])

            highlowpass_shearlets.append(highlowpass_shearlet)

            lowhighpass_shearlet = preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, -shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone1"]["lowhighpass"])

            lowhighpass_shearlets.append(lowhighpass_shearlet)
        else:
            """            
            shearlets[:, :, j] = np.transpose(
                preparedFilters["cone2"]["wedge"][
                    preparedFilters["shearLevels"][scale - 1]
                ][
                    :,
                    :,
                    shearing + np.power(2, preparedFilters["shearLevels"][scale - 1]),
                ]
                * np.conj(preparedFilters["cone2"]["bandpass"][:, :, scale - 1])
            )
            """
            bandpass_shearlet = np.transpose(preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone2"]["bandpass"])
            )

            bandpass_shearlets.append(bandpass_shearlet)         
            
            lowpass_shearlet = np.transpose(preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone2"]["lowpass"])
            )

            lowpass_shearlets.append(lowpass_shearlet)            
            
            highlowpass_shearlet = np.transpose(preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone2"]["highlowpass"])
            )

            highlowpass_shearlets.append(highlowpass_shearlet)

            lowhighpass_shearlet = np.transpose(preparedFilters["cone1"]["wedge"][
                preparedFilters["shearLevels"][scale - 1]
            ][
                :, :, shearing + np.power(2, preparedFilters["shearLevels"][scale - 1])
            ] * np.conj(preparedFilters["cone2"]["highlowpass"])
            )

            lowhighpass_shearlets.append(lowhighpass_shearlet)
        
    print(len(lowpass_shearlets), len(bandpass_shearlets))
    interleaved = sum([[high, low] for high, low in zip(bandpass_shearlets, lowpass_shearlets)], [])
    shearlets = np.stack(interleaved, -1)
    shearlets_low = np.stack(lowpass_shearlets[:-1], -1)
    shearlets_high = np.stack(bandpass_shearlets, -1)

    RMS = np.linalg.norm(shearlets, axis=(0, 1)) / np.sqrt(rows * cols)
    dualFrameWeights= np.sum(np.power(np.abs(shearlets), 2), axis=-1)
    dualFrameWeights_high = np.sum(np.power(np.abs(shearlets_high), 2), axis=-1)
    dualFrameWeights_low = np.sum(np.power(np.abs(shearlets_low), 2), axis=-1)

    return shearlets_high, RMS, dualFrameWeights_high, shearlets_low, RMS, dualFrameWeights_low


def myShearletSystem(    
    rows,
    cols,
    shearLevels=None,
    full=0,
    directionalFilter=None,
    quadratureMirrorFilter=None,):
    nScales = 1
    # check which args are given and set default values if necccessary
    if shearLevels is None:
        shearLevels = np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
    if directionalFilter is None:
        h0, h1 = dfilters("dmaxflat4", "d")

        h0 /= np.sqrt(2)
        h1 /= np.sqrt(2)

        directionalFilter = modulate2(h0, "c")
    if quadratureMirrorFilter is None:
        quadratureMirrorFilter = np.array(
            [
                0.0104933261758410,
                -0.0263483047033631,
                -0.0517766952966370,
                0.276348304703363,
                0.582566738241592,
                0.276348304703363,
                -0.0517766952966369,
                -0.0263483047033631,
                0.0104933261758408,
            ]
        )
    # skipping use gpu stuff for the moment...
    preparedFilters = prepareFilters(
        rows, cols, shearLevels, directionalFilter, quadratureMirrorFilter
    )
    shearletIdxs = SLgetShearletIdxs2D(shearLevels, full)
    shearlets_high, RMS_high, dualFrameWeights_high, shearlets_low, RMS_low, dualFrameWeights_low = getShearlets(preparedFilters, shearletIdxs)

    # create dictionary
    shearletSystem_high = {
        "shearlets": shearlets_high,
        "size": preparedFilters["size"],
        "shearLevels": preparedFilters["shearLevels"],
        "full": full,
        "nShearlets": shearlets_high.shape[-1],
        "shearletIdxs": shearletIdxs,
        "dualFrameWeights": dualFrameWeights_high,
        "RMS": RMS_high,
        "useGPU": 0,
        "isComplex": 0,
        "wedge": preparedFilters["cone1"]["wedge"]
    }
    
    shearletSystem_low = {
        "shearlets": shearlets_low,
        "size": preparedFilters["size"],
        "shearLevels": preparedFilters["shearLevels"],
        "full": full,
        "nShearlets": shearlets_low.shape[-1],
        "shearletIdxs": shearletIdxs,
        "dualFrameWeights": dualFrameWeights_low,
        "RMS": RMS_low,
        "useGPU": 0,
        "isComplex": 0,
    }
    return shearletSystem_high, shearletSystem_low  



class ShearletModule(torch.nn.Module):
    def __init__(self, filters=1, levels=1):
        # first we need to do a one-time generation of the wedge
        self.wedge = self.prepare_wedge(levels)
    
        # then we need to define a parameter that we will use to store the kernel for the mother shearlet
        # eventually we might want to support multiple filters (TODO)
        # because we are learning here we will not reverse these guys across the time axis as that is meaningless.
        # technically not reversing across the time axis makes this cross-correlation but it is what it is.
        self.filter = self.prepare_filters()

    def prepare_wedge(self, levels):
        pass


    def getPassFilts(
        rows,
        cols,
        shearLevels,
        scalingFilter=None,
        ):

        if scalingFilter is None:
            # TODO: this was a choice so it should be moved to init
            scalingFilter = np.array(
                [
                    0.0104933261758410,
                    -0.0263483047033631,
                    -0.0517766952966370,
                    0.276348304703363,
                    0.582566738241592,
                    0.276348304703363,
                    -0.0517766952966369,
                    -0.0263483047033631,
                    0.0104933261758408,
                ]
            )

        wedge = [None] * (
            max(shearLevels) + 1
        )  # these filters partition the frequenecy plane into different directions

        filterLow2 = [None] * (max(shearLevels) + 1)

        filterLow2[-1] = scalingFilter

        # TODO: this was a choice so it should be moved to init
        h0, _ = dfilters("dmaxflat4", "d") 
        h0 /= np.sqrt(2)
        directionalFilter = modulate2(h0, "c")

        # normalize filters
        # TODO: double sum is ugly - replace
        directionalFilter = directionalFilter / sum(sum(np.absolute(directionalFilter)))


        filterLow2[-1].shape = (1, len(filterLow2[-1]))
        for shearLevel in np.unique(shearLevels):

            wedge[shearLevel] = np.zeros(
                (rows, cols, int(np.floor(np.power(2, shearLevel + 1) + 1))), dtype=complex
            )

            directionalFilterUpsampled = SLupsample(
                directionalFilter, 1, np.power(2, shearLevel + 1) - 1
            )

            filterLow2[-1 - shearLevel].shape = (1, len(filterLow2[-1 - shearLevel]))

            wedgeHelp = scipy.signal.convolve2d(
                directionalFilterUpsampled,
                np.transpose(filterLow2[len(filterLow2) - shearLevel - 1]),
            )
            wedgeHelp = SLpadArray(wedgeHelp, np.array([rows, cols]))

            wedgeUpsampled = SLupsample(wedgeHelp, 2, np.power(2, shearLevel) - 1)

            lowpassHelp = SLpadArray(
                filterLow2[len(filterLow2) - max(shearLevel - 1, 0) - 1],
                np.asarray(wedgeUpsampled.shape),
            )
            if shearLevel >= 1:
                wedgeUpsampled = fftlib.fftshift(
                    fftlib.ifft2(
                        fftlib.ifftshift(
                            fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(lowpassHelp)))
                            * fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(wedgeUpsampled)))
                        )
                    )
                )
            lowpassHelpFlip = np.fliplr(lowpassHelp)
            # traverse all directions of the upper part of the left horizontal
            # frequency cone
            for k in range(-np.power(2, shearLevel), np.power(2, shearLevel) + 1):
                # resample wedgeUpsampled as given in equation (22) on page 15.
                wedgeUpsampledSheared = SLdshear(wedgeUpsampled, k, 2)
                # convolve again with flipped lowpass filter, as required by
                # equation (22) on page 15
                if shearLevel >= 1:
                    wedgeUpsampledSheared = fftlib.fftshift(
                        fftlib.ifft2(
                            fftlib.ifftshift(
                                fftlib.fftshift(
                                    fftlib.fft2(fftlib.ifftshift(lowpassHelpFlip))
                                )
                                * fftlib.fftshift(
                                    fftlib.fft2(fftlib.ifftshift(wedgeUpsampledSheared))
                                )
                            )
                        )
                    )
                # obtain downsampled and renormalized and sheared wedge filter
                # in the frequency domain, according to equation (22), page 15.
                wedge[shearLevel][:, :, int(np.fix(np.power(2, shearLevel)) - k)] = (
                    fftlib.fftshift(
                        fftlib.fft2(
                            fftlib.ifftshift(
                                np.power(2, shearLevel)
                                * wedgeUpsampledSheared[
                                    :,
                                    0 : np.power(2, shearLevel) * cols - 1 : np.power(
                                        2, shearLevel
                                    ),
                                ]
                            )
                        )
                    )
                )

        return wedge


    def prepareFilters(
        self,
        rows,
        cols,
        shearLevels=None,
        directionalFilter=None,
        quadratureMirrorFilter=None,
        ):

        nScales=1,
        scalingFilter = quadratureMirrorFilter
        scalingFilter2, waveletFilter = None, None
        # check input arguments
        if shearLevels is None:
            shearLevels = np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
        if scalingFilter is None:
            scalingFilter = np.array(
                [
                    0.0104933261758410,
                    -0.0263483047033631,
                    -0.0517766952966370,
                    0.276348304703363,
                    0.582566738241592,
                    0.276348304703363,
                    -0.0517766952966369,
                    -0.0263483047033631,
                    0.0104933261758408,
                ]
            )
        if scalingFilter2 is None:
            scalingFilter2 = scalingFilter
        if directionalFilter is None:
            h0, h1 = dfilters("dmaxflat4", "d")
            h0 /= np.sqrt(2)
            h1 /= np.sqrt(2)
            directionalFilter = modulate2(h0, "c")
        if waveletFilter is None:
            waveletFilter = MirrorFilt(scalingFilter)

        fSize = np.array([rows, cols])
        filters = {"size": fSize, "shearLevels": shearLevels}
        wedge, bandpass, highlowpass, lowhighpass, lowpass = getPassFilts(
            rows,
            cols,
            shearLevels,
            scalingFilter,
        )
        wedge[0] = 0  # for matlab compatibilty (saving filters as .mat files)
        filters["cone1"] = {"wedge": wedge, "bandpass": bandpass, "lowpass": lowpass, "highlowpass": highlowpass, "lowhighpass": lowhighpass}
        if rows == cols:
            filters["cone2"] = filters["cone1"]
        else:
            wedge2, bandpass, highlowpass, lowhighpass, lowpass = getPassFilts(
                cols,
                rows,
                shearLevels,
                scalingFilter,
            )
            wedge2[0] = 0  # for matlab compatibilty (saving filters as .mat files)
            filters["cone2"] = {"wedge": wedge2, "bandpass": bandpass, "lowpass": lowpass, "highlowpass": highlowpass, "lowhighpass": lowhighpass}

        return filters
    
    def forward(self):
        """
        perform the forward pass of the operation

        note that when shearing the mother it is probably faster to fft and conv at higher kernel sizes and resolutions (?)
        this is definitely true at the highest resolutions, but probably untrue over all at the lowest resolutions. we will 
        need to investigate which is fastest, though it is unclear how to apply the complex modulus non-linearity if we do 
        not.

        """
        pass