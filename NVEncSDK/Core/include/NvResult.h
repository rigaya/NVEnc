/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


//---------------------------------------------------------------------------
//! \file NvResult.h
//! \brief Method and function results.
//!
//! Most methods and functions used in this SDK use NvResult as a return type.
//! This file contains the definition of the NvResult enumeration and various
//! facility functions that deal with the NvResult type.
//---------------------------------------------------------------------------

#ifndef _NV_RESULT_H
#define _NV_RESULT_H

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <include/NvResultTable.h>

//! \brief Cause a machine access violation.
//! This will halt the executing program. Use if a fatal error is detected.
// However, ignore the intentional faults during Coverity Prevent analysis.
#ifdef __COVERITY__
#define NV_ACCESS_VIOLATION
#else
#define NV_ACCESS_VIOLATION \
    { volatile int *p = 0; volatile int n = *p; p += n; }
#endif

//! Declares the \e NvResult enumerations.
#define NV_RESULT_ENTRY(ENUM, DESCRIPTION) \
    ENUM,

//! Declares the \e NvResult enumerations.
#define NV_RESULT_ENTRY_VALUE(ENUM, VALUE, DESCRIPTION) \
    ENUM = VALUE,

//! \brief Return result of methods used by most interfaces.
//!
//! Most methods declared in the public interfaces will return \e NvResult.
//! This emum contains all the possible \e success and \e failure return values.
//! If \e NvResult is equal or greater than zero the result is \e success.
//! If \e NvResult is less than zero the result is \e failure. Use the
//! \e IsFailed() and \e IsSucceeded() functions to evaluate the return code.
//! The return codes themselves are defined in NvResultTable.h.
enum NvResult
{
    //! See NvResultTable.h for the list of enumeration values.
    NV_RESULT_TABLE
    RESULT_NUM
};

//! \brief true if eResult represents success.
//! \param eResult Interface method or function result to evaulate.
//! \return Indicates whether eResult is a success.
//! \retval true eResult represents success.
//! \retval false eResult represents failure.
inline bool IsSucceeded(NvResult eResult)
{
    return (eResult & 0x80000000) == 0;
}

//! \brief execute function and return error on failure
//! \param function Method or function to execute
//! \retval true \e function returned failure.
//! \retval false \e function returned success.
#define NvReturnError(function) \
    er = function;                \
    if (IsFailed(er)) return er;

//! \brief true if eResult represents failure.
//! \param eResult Interface method or function result to evaulate.
//! \return Indicates whether eResult is a failure.
//! \retval true eResult represents failure.
//! \retval false eResult represents success.
//! \code
//!
//! NvResult er = spClass->Method();
//! if (IsFailed(er)) {
//!     // Handle or log the error.
//!     return er;
//! }
//! \endcode
inline bool IsFailed(NvResult eResult)
{
    return (eResult & 0x80000000) != 0;
}

//! \brief Accumulate error codes while maintaining the error bit.
//! \param er1 Original error code (accumulator).
//! \param er2 New NvResult to accumulate.
//! \return NvResult cast integer to indicate whether the error bit is set.
//! \retval Greater than or equal to zero for success.
//! \retval Less than zero for failure.
//!
//! \code
//! NvResult er = spClass->Method1();
//! er |= spClass->Method2();
//! er |= spClass->Method3();
//! if (IsFailed(er)) {
//!     // Handle or log the undefined error.
//!     return er;
//! }
//! \endcode
//! \warning When this function is used the meaning of the left
//! NvResult is undefined. Use the \e operator|= to detect a failure
//! in a list of calls. You cannot determine which call failed or
//! what the failure code was, just the fact that something failed.
//! Compare this function with NvCheckError() and use the one
//! appropriate for your situation.
inline NvResult operator|=(NvResult &er1, const NvResult &er2)
{
    return er1 = static_cast<NvResult>(static_cast<int>(er1) | static_cast<int>(er2));
}

//! \brief Update current eCurrentResult with a new eNewResult.
//! \param eCurrentResult current result of previous calls to \e NvCheckError().
//! \param eNewResult new result.
//! \return Returns whether an error has been found.
//! \retval true error detected.
//! \retval false no errors detected.
//!
//! This function will update eCurrentResult with eNewResult if eNewResult
//! is a failure and eCurrentResult doesn't already contain a failure.
//! This function can be used to maintain the failure code of the first
//! failure encountered even if subsequent calls fail.
//! \code
//! NvResult er = spClass->Method1();
//! NvCheckError(er, spClass->Method2());
//! NvCheckError(er, spClass->Method3());
//! if (IsFailed(er)) {
//!     // Handle or log the first error found.
//!     return er;
//! }
//! \endcode
inline bool NvCheckError(NvResult &eCurrentResult, NvResult eNewResult)
{
    eCurrentResult = IsFailed(eCurrentResult) ? eCurrentResult : eNewResult;
    return IsFailed(eCurrentResult);
}

//! \brief Halt exection if \e eResult represents a failure.
//! \param eResult Result to check.
//! \param pszFile Source filename where check is being made.
//! \param nLine Line number where check is being made.
//! \warning If the code was compiled for production, the value of \e eResult
//! is ignored and the assert will not stop execution.
inline void NvAssertSucceeded(NvResult eResult, const char *pszFile, int nLine)
{
    if (IsFailed(eResult))
    {
#ifndef NV_PRODUCTION_BUILD // Define for builds that shouldn't assert or fault on failure
        assert("AssertSucceeded Failed- See errors.log for specific info" && 0);
        NV_ACCESS_VIOLATION
#endif
    }
}

//! Halt execution if \e eResult represents a failure.
//! \see \e NvAssertSucceeded.
#define AssertSucceeded(eResult) \
    NvAssertSucceeded(eResult, __FILE__, __LINE__)

#undef NV_RESULT_ENTRY_VALUE
#undef NV_RESULT_ENTRY

#endif
