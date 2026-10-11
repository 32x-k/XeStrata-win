// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Deliberately no real Win32 includes/imports: compile direct_file.cpp against this header.
#pragma once
#include <cstddef>
#include <cstdint>

using BOOL = int;
using DWORD = std::uint32_t;
using ULONG = std::uint32_t;
using ULONG_PTR = std::uintptr_t;
using HANDLE = void*;
struct OVERLAPPED {
    ULONG_PTR Internal = 0;
    ULONG_PTR InternalHigh = 0;
    DWORD Offset = 0;
    DWORD OffsetHigh = 0;
    HANDLE hEvent = nullptr;
};
struct OVERLAPPED_ENTRY {
    ULONG_PTR lpCompletionKey;
    OVERLAPPED* lpOverlapped;
    ULONG_PTR Internal;
    DWORD dwNumberOfBytesTransferred;
};
struct LARGE_INTEGER { std::int64_t QuadPart; };
#define INVALID_HANDLE_VALUE reinterpret_cast<HANDLE>(static_cast<std::intptr_t>(-1))
constexpr BOOL TRUE = 1, FALSE = 0;
constexpr DWORD INFINITE = 0xffffffffu;
constexpr DWORD ERROR_IO_PENDING = 997;
constexpr DWORD ERROR_HANDLE_EOF = 38, ERROR_ACCESS_DENIED = 5;
constexpr DWORD ERROR_NOT_FOUND = 1168, ERROR_GEN_FAILURE = 31, ERROR_OPERATION_ABORTED = 995;
constexpr DWORD WAIT_TIMEOUT = 258;
constexpr DWORD MEM_RESERVE = 0x2000, MEM_COMMIT = 0x1000, MEM_RELEASE = 0x8000;
constexpr DWORD PAGE_READWRITE = 4, CP_UTF8 = 65001, GENERIC_READ = 0x80000000;
constexpr DWORD FILE_SHARE_READ = 1, OPEN_EXISTING = 3;
constexpr DWORD FILE_FLAG_NO_BUFFERING = 0x20000000, FILE_FLAG_OVERLAPPED = 0x40000000;
constexpr DWORD FILE_FLAG_RANDOM_ACCESS = 0x10000000;

void* VirtualAlloc(void*, std::size_t, DWORD, DWORD);
BOOL VirtualFree(void*, std::size_t, DWORD);
int MultiByteToWideChar(unsigned, DWORD, const char*, int, wchar_t*, int);
HANDLE CreateFileW(const wchar_t*, DWORD, DWORD, void*, DWORD, DWORD, HANDLE);
BOOL GetFileSizeEx(HANDLE, LARGE_INTEGER*);
HANDLE CreateIoCompletionPort(HANDLE, HANDLE, ULONG_PTR, DWORD);
BOOL CloseHandle(HANDLE);
DWORD GetLastError();
BOOL ReadFile(HANDLE, void*, DWORD, DWORD*, OVERLAPPED*);
BOOL PostQueuedCompletionStatus(HANDLE, DWORD, ULONG_PTR, OVERLAPPED*);
BOOL GetQueuedCompletionStatusEx(HANDLE, OVERLAPPED_ENTRY*, ULONG, ULONG*, DWORD, BOOL);
BOOL GetQueuedCompletionStatus(HANDLE, DWORD*, ULONG_PTR*, OVERLAPPED**, DWORD);
BOOL CancelIoEx(HANDLE, OVERLAPPED*);
