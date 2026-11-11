// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Keep the optional fp16 kernels apart from the portable light kernels.
#define STRATA_DSV41_NATIVE_F16_OUTPUT 1
#include "ds4_iq.cpp"  // NOLINT(bugprone-suspicious-include): share kernel bodies with the portable image.
