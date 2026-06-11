"""Helpers for the submission notebook source package."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def reload_submission_src() -> list[str]:
	"""Reload every imported module under ``submission_folder.src``.

	This is useful in notebooks where files under ``submission_folder/src/``
	are edited repeatedly and the kernel should pick up changes without a
	restart.
	"""

	package_name = __name__
	package = importlib.import_module(package_name)

	module_names = {package_name}
	if hasattr(package, "__path__"):
		module_names.update(
			module_info.name
			for module_info in pkgutil.walk_packages(package.__path__, package_name + ".")
		)

	reloaded_modules: list[str] = []
	for module_name in sorted(module_names, key=len, reverse=True):
		module = importlib.import_module(module_name)
		importlib.reload(module)
		reloaded_modules.append(module_name)

	return reloaded_modules
