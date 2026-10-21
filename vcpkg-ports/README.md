# vcpkg overlay ports

Ports here replace the ones at `builtin-baseline` in `vcpkg.json`. Each is a
verbatim copy of a newer upstream port; delete it once the baseline moves to
a vcpkg release that contains that version.

- `rubberband` 4.0.0#2, from microsoft/vcpkg 173a2013: adds
  `fix-size-t.patch` (breakfastquay/rubberband#131), without which rubberband
  doesn't compile with Xcode 26.6's libc++. The `2026.07.29` release has
  4.0.0#1.
