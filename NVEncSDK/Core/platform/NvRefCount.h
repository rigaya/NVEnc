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
//! \file NvRefCount.h
//! \brief Base abstract interface for all reference counted classes.
//---------------------------------------------------------------------------

#ifndef _NV_IREFCOUNT_ITF_H
#define _NV_IREFCOUNT_ITF_H

#include "NvCallingConventions.h"

//! \brief Base abstract interface for all reference counted classes.

//! \e INvRefCount is the base abstract interface for all reference counted classes.
//! Each class that inherits from \e INvRefCount must implement \e AddRef()
//! and \e Release(). When a reference counted class is created, the initial
//! reference count will be 1. Whenever a new variable points to that class
//! the application should call \e AddRef() after assignment. Each time the
//! variable is destroyed, \e Release() should be called. On each \e AddRef()
//! the reference count is incremented. On each \e Release() the reference
//! count is decremented. When the reference count goes to 0 the object is
//! automatically destroyed.
//!
//! An implementation of INvRefCount is provided by \e INvRefCountImpl.
//! Application developers will typically use \e INvRefCountImpl instead of
//! deriving new objects from \e INvRefCount.
class INvRefCount
{
    public:

        //!
        //! Increment the reference count by 1.
        virtual unsigned long NV_CALL_CONV_COM AddRef() = 0;

        //! Decrement the reference count by 1. When the reference count
        //! goes to 0 the object is automatically destroyed.
        virtual unsigned long NV_CALL_CONV_COM Release() = 0;

    protected:

        virtual ~INvRefCount() { }
};

//! \name Disable copy and assign macros
//! The following macros disable some default standard C++ behaviour that
//! can cause trouble for reference counted classes. When the standard features
//! are used by mistake the reference count can become inaccurate or meaningless.
//@{

//! Disable the default copy constructor.
#define DISABLE_COPY_CONSTRUCTOR(type) protected: type(const type &inst) { }

//! Disable the default copy constructor for a derived base.
#define DISABLE_COPY_CONSTRUCTOR_WITH_BASE(type, base) protected: type(const type &inst) : base { }

//! Disable the default assignment operator.
#define DISABLE_ASSIGNMENT(type) protected: type & operator=(const type &inst) { return *this; }

//! Disable the default copy constructor and assignment operator.
#define DISABLE_COPY_AND_ASSIGN(type) DISABLE_COPY_CONSTRUCTOR(type) DISABLE_ASSIGNMENT(type)

//! Disable the default copy constructor and assignment operator for a derived base.
#define DISABLE_COPY_AND_ASSIGN_WITH_BASE(type, base) DISABLE_COPY_CONSTRUCTOR_WITH_BASE(type, base) DISABLE_ASSIGNMENT(type)

//@}
#endif
