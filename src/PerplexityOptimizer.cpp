////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2008, Massachusetts Institute of Technology              //
// All rights reserved.                                                   //
//                                                                        //
// Redistribution and use in source and binary forms, with or without     //
// modification, are permitted provided that the following conditions are //
// met:                                                                   //
//                                                                        //
//     * Redistributions of source code must retain the above copyright   //
//       notice, this list of conditions and the following disclaimer.    //
//                                                                        //
//     * Redistributions in binary form must reproduce the above          //
//       copyright notice, this list of conditions and the following      //
//       disclaimer in the documentation and/or other materials provided  //
//       with the distribution.                                           //
//                                                                        //
//     * Neither the name of the Massachusetts Institute of Technology    //
//       nor the names of its contributors may be used to endorse or      //
//       promote products derived from this software without specific     //
//       prior written permission.                                        //
//                                                                        //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      //
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  //
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT   //
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  //
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT       //
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  //
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  //
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT    //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  //
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   //
////////////////////////////////////////////////////////////////////////////

#include <ctime>
#include "util/Logger.h"
#include "PerplexityOptimizer.h"
#include "util/ZFile.h"
#define MAXLINE 65535


////////////////////////////////////////////////////////////////////////////////

void
PerplexityOptimizer::LoadCorpus(ZFile &corpusFile) {
    //const CountVector &counts(_lm.counts(1));
    //BitVector vocabMask = (_lm.counts > 0);
    BitVector vocabMask(_lm.vocab().size(), 1);
    _lm._pModel->LoadEvalCorpus(_probCountVectors, _bowCountVectors,
                                vocabMask, corpusFile, _numOOV, _numWords);

    vector<BitVector> probMaskVectors(_order + 1);
    vector<BitVector> bowMaskVectors(_order);
    for (size_t o = 0; o <= _order; o++)
        probMaskVectors[o] = (_probCountVectors[o] > 0);
    for (size_t o = 0; o < _order; o++)
        bowMaskVectors[o] = (_bowCountVectors[o] > 0);
    _mask = _lm.GetMask(probMaskVectors, bowMaskVectors);
}

double
PerplexityOptimizer::ShortCorpusComputeEntropy(ZFile &corpusFile, const ParamVector &params) {
    if (corpusFile == NULL) throw std::invalid_argument("Invalid file");
    size_t size = _lm._pModel->size();
//     BitVector vocabMask(size, 1);
    // Accumulate counts of prob/bow for computing perplexity of corpusFilename.
    char                    line[MAXLINE];
    size_t                  numOOV = 0;
    vector<VocabIndex> words(256);
    _totLogProb = 0.0;
    _numZeroProbs = 0;
    size_t numWords = 0;
    size_t numZeroProbs = 0;
    double lp;
    double unkLogProb = 70.0;
//     size_t total_unigrams = sum(((NgramLM&)_lm).counts(1));
//      unkLogProb = log((double)(_lm.vocab().size()))-log((double)total_unigrams);
//     unkLogProb = -log((double)(_lm.vocab().size()));
//     std::cerr << _lm.vocab().size() << "\t" << total_unigrams << "\t" << unkLogProb << std::endl;
//     exit(1);
    while (corpusFile.getLine( line, MAXLINE)) {
        if (strncmp(line, "<DOC ", 5) == 0 || strcmp(line, "</DOC>") == 0)
            continue;
//      Logger::Log(0, "Additional Input:%s\n", line);
        // Lookup vocabulary indices for each word in the line.
        words.clear();
//         words.push_back(Vocab::EndOfSentence);
        char *p = &line[0];
        while (*p != 0) {
            while (isspace(*p)) ++p;  // Skip consecutive spaces.
            const char *token = p;
            while (*p != 0 && !isspace(*p))  ++p;
            size_t len = p - token;
            if (*p != 0) *p++ = 0;
            words.push_back(_lm.vocab().Find(token, len));
        }
//         words.push_back(Vocab::EndOfSentence);

        // Add each top order n-gram.
        size_t ngramOrder = std::min((size_t)1, size - 1);
        for (size_t i = 0; i < words.size(); i++) {
            if (words[i] == Vocab::Invalid) {
                // OOV word encountered.  Reset order to unigrams.
                ngramOrder = 1;
                numOOV++;
                return 74e70;
            } else {
                NgramIndex index;
                size_t     boOrder = ngramOrder;
                while ((index = _lm._pModel->_Find(&words[i-boOrder+1], boOrder)) == -1) {
                    --boOrder;
                    NgramIndex hist = _lm._pModel->_Find(&words[i - boOrder], boOrder);
                    if (hist != (NgramIndex)-1) {
                        if ((_lm.bows(boOrder))[hist] != 0) {
//                           _bowCountVectors[boOrder][hist]++;
                            lp = log((_lm.bows(boOrder))[hist]);
                            _totLogProb += lp * 1;
//                             std::cerr << hist
//                               << "\t" << (_lm.bows(boOrder))[hist]
//                               << "\tb "
//                               << lp * 1 
//                               << "\t" << boOrder 
//                               << "\t" << i
//                               << "\t" << i - boOrder
//                               << "\n";
                        } else {
                          return 75e70;
                        }
                    }
                }
                ngramOrder = std::min(ngramOrder + 1, size - 1);
//                 _probCountVectors[boOrder][index]++;
                if ((_lm.probs(boOrder))[index] == 0 ) {
                    numWords++;
                    numZeroProbs++;
                    return 77e70;
                } else if (words[i] == Vocab::EndOfSentence) {
                    (void)0;
                } else {
                    lp = log((_lm.probs(boOrder))[index]) * 1;
//                     std::cerr << index
//                       << "\t" << (_lm.probs(boOrder))[index]
//                       << "\t" << (((NgramLM&)_lm).counts(boOrder))[index]
//                       << "\tp "
//                       << lp * 1 
//                       << "\t" << boOrder 
//                       << "\t" << ngramOrder 
//                       << "\n";
                    if (!std::isfinite(lp)) {
                      std::cerr << lp << "\t" << (_lm.probs(boOrder))[index]  << "\t" << boOrder << "\t" << index << " " << (_lm.probs(boOrder)).length() << "\t" << ngramOrder << " " << size << std::endl;
                      return 72e70;
                    }
                    _totLogProb += lp;
                    if (!std::isfinite(_totLogProb)) {
                      std::cerr << -_totLogProb << "\t" << numWords << "\t" << numZeroProbs << std::endl;
                      return 71e70;
                    }
                    numWords++;
                }
//                 std::cerr << numWords << "\n\n";
            }
        }
    }
    if ((numWords - numZeroProbs) < 1) {
        return 76e70;
    }
//     double entropy = -_totLogProb / (numWords - numZeroProbs);
    double entropy = -_totLogProb;
//     std::cerr 
//       << -_totLogProb 
//       << "\t" << numWords 
//       << "\t" << numZeroProbs 
//       << std::endl;
    if (!std::isfinite(entropy)) {
        std::cerr << -_totLogProb << "\t" << numWords << "\t" << numZeroProbs << std::endl;
    }
    if (Logger::GetVerbosity() > 2)
        std::cout << exp(entropy) << "\t" << params << std::endl;
    else
        Logger::Log(2, "%f\n", exp(entropy));
    return (std::isfinite(entropy)) ? entropy : 73e70;
}

bool
PerplexityOptimizer::EstimateOnly(const ParamVector &params) {
    // Estimate model.
    return _lm.Estimate(params, _mask);
}

double
PerplexityOptimizer::ComputeEntropyNoEstimate(const ParamVector &params) {

    // Compute total log probability and num zero probs.
    _totLogProb = 0.0;
    _numZeroProbs = 0;
    for (size_t o = 0; o <= _order; o++) { 
        // assert(alltrue(counts == 0 || probs > 0));
        // _totLogProb += dot(log(probs), counts, counts > 0);
        // _totLogProb += sum((log(probs) * counts)[counts > 0]);
        const CountVector &counts(_probCountVectors[o]);
        const ProbVector & probs(_lm.probs(o));
        for (size_t i = 0; i < counts.length(); i++) {
            if (counts[i] > 0) {
                assert(std::isfinite(probs[i]));
                if (probs[i] == 0)
                    _numZeroProbs++;
                else
                    _totLogProb += log(probs[i]) * counts[i];
            }
        }
    }
    for (size_t o = 0; o < _order; o++) {
        // assert(allTrue(counts == 0 || bows > 0));
        // _totLogProb += dot(log(bows), counts, counts > 0);
        const CountVector &counts(_bowCountVectors[o]);
        const ProbVector & bows(_lm.bows(o));
        for (size_t i = 0; i < counts.length(); i++) {
            if (counts[i] > 0) {
                assert(std::isfinite(bows[i]));
                assert(bows[i] != 0);
                if (bows[i] == 0)
                    Logger::Warn(1, "Invalid BOW %lu %lu %i\n", o,i,counts[i]);
                _totLogProb += log(bows[i]) * counts[i];
            }
        }
    }

    double entropy = -_totLogProb / (_numWords - _numZeroProbs);
//     double entropy = -_totLogProb / _numWords;
    if (Logger::GetVerbosity() > 2)
        std::cout << exp(entropy) << "\t" << params << std::endl;
    else
        Logger::Log(2, "%f\n", exp(entropy));
    return std::isnan(entropy) ? 70 : entropy;
}


double
PerplexityOptimizer::ComputeEntropy(const ParamVector &params) {
    // Estimate model.
    if (!_lm.Estimate(params, _mask))
        return 70;  // Out of bounds.  Corresponds to perplexity = 1100.

    // Compute total log probability and num zero probs.
    _totLogProb = 0.0;
    _numZeroProbs = 0;
    for (size_t o = 0; o <= _order; o++) { 
        // assert(alltrue(counts == 0 || probs > 0));
        // _totLogProb += dot(log(probs), counts, counts > 0);
        // _totLogProb += sum((log(probs) * counts)[counts > 0]);
        const CountVector &counts(_probCountVectors[o]);
        const ProbVector & probs(_lm.probs(o));
        for (size_t i = 0; i < counts.length(); i++) {
            if (counts[i] > 0) {
                assert(std::isfinite(probs[i]));
                if (probs[i] == 0)
                    _numZeroProbs++;
                else
                    _totLogProb += log(probs[i]) * counts[i];
            }
        }
    }
    for (size_t o = 0; o < _order; o++) {
        // assert(allTrue(counts == 0 || bows > 0));
        // _totLogProb += dot(log(bows), counts, counts > 0);
        const CountVector &counts(_bowCountVectors[o]);
        const ProbVector & bows(_lm.bows(o));
        for (size_t i = 0; i < counts.length(); i++) {
            if (counts[i] > 0) {
                assert(std::isfinite(bows[i]));
                assert(bows[i] != 0);
                if (bows[i] == 0)
                    Logger::Warn(1, "Invalid BOW %lu %lu %i\n", o,i,counts[i]);
                _totLogProb += log(bows[i]) * counts[i];
            }
        }
    }

    double entropy = -_totLogProb / (_numWords - _numZeroProbs);
    if (Logger::GetVerbosity() > 2)
        std::cout << exp(entropy) << "\t" << params << std::endl;
    else
        Logger::Log(2, "%f\n", exp(entropy));
    return std::isnan(entropy) ? 70 : entropy;
}

double
PerplexityOptimizer::Optimize(ParamVector &params, Optimization technique) {
    _numCalls = 0;
    ComputeEntropyFunc func(*this);
    int     numIter;
    double  minEntropy;
    clock_t startTime = clock();
    switch (technique) {
    case PowellOptimization:
        minEntropy = MinimizePowell(func, params, numIter);
        break;
    case LBFGSOptimization:
        minEntropy = MinimizeLBFGS(func, params, numIter);
        break;
    case LBFGSBOptimization:
        minEntropy = MinimizeLBFGSB(func, params, numIter);
        break;
    default:
        throw std::runtime_error("Unsupported optimization technique.");
    }
    clock_t endTime = clock();

    Logger::Log(1, "Iterations    = %i\n", numIter);
    Logger::Log(1, "Elapsed Time  = %f\n",
                float(endTime - startTime) / CLOCKS_PER_SEC);
    Logger::Log(1, "Perplexity    = %f\n", exp(minEntropy));
    Logger::Log(1, "Num OOVs      = %lu\n", _numOOV);
    Logger::Log(1, "Num ZeroProbs = %lu\n", _numZeroProbs);
    Logger::Log(1, "Func Evals    = %lu\n", _numCalls);
    Logger::Log(1, "OptParams     = [ ");
    for (size_t i = 0; i < params.length(); i++)
        Logger::Log(1, "%f ", params[i]);
    Logger::Log(1, "]\n");
    return minEntropy;
}
