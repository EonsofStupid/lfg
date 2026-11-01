"""The SPDX image root must not hide unexpected container dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


path = Path(__file__).resolve().parents[2] / 'scripts' / 'validate-sbom-policy.py'
spec = importlib.util.spec_from_file_location('sbom_policy', path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def package(identifier, purl, purpose=None):
    return {'SPDXID': identifier, 'primaryPackagePurpose': purpose,
            'externalRefs': [{'referenceLocator': purl}]}


class SbomDocumentRootTests(unittest.TestCase):
    def setUp(self):
        self.sbom = {'SPDXID': 'SPDXRef-DOCUMENT',
                     'relationships': [{'spdxElementId': 'SPDXRef-DOCUMENT', 'relationshipType': 'DESCRIBES',
                                        'relatedSpdxElement': 'SPDXRef-root'}],
                     'packages': [package('SPDXRef-root', 'pkg:oci/lfg@sha256%3A123', 'CONTAINER'),
                                  package('SPDXRef-python', 'pkg:pypi/example@1')]}
        self.allowed = {'oci': {'<none>'}}

    def kinds(self, sbom):
        return subject._collect_package_kinds(sbom, [], self.allowed)

    def test_exact_configured_container_root_is_not_treated_as_dependency(self):
        self.assertEqual(self.kinds(self.sbom), [('pypi', '<none>')])
        self.assertEqual(subject._count_packages(self.sbom), 2)
        self.assertIn(('oci', '<none>'), subject._collect_package_kinds(self.sbom, []))

    def test_nested_oci_package_is_still_reported(self):
        self.sbom['packages'].append(package('SPDXRef-nested', 'pkg:oci/other@sha256%3A456', 'CONTAINER'))
        self.assertIn(('oci', '<none>'), self.kinds(self.sbom))

    def test_policy_cli_accepts_only_the_configured_root(self):
        root = path.parents[1]
        with tempfile.TemporaryDirectory() as directory:
            sbom_path = Path(directory) / 'sbom.json'
            args = [sys.executable, str(path), '--sbom', str(sbom_path),
                    '--policy', str(root / 'security' / 'sbom-policy.json')]
            sbom_path.write_text(json.dumps(self.sbom), encoding='utf-8')
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.sbom['packages'].append(package('SPDXRef-nested', 'pkg:oci/other@sha256%3A456', 'CONTAINER'))
            sbom_path.write_text(json.dumps(self.sbom), encoding='utf-8')
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)

    def test_ambiguous_or_unconfigured_roots_are_not_exempt(self):
        for change in ('no-description', 'two-descriptions', 'wrong-document', 'duplicate-id',
                       'multiple-purls', 'wrong-purpose', 'wrong-namespace'):
            sbom = copy.deepcopy(self.sbom)
            if change == 'no-description':
                sbom['relationships'] = []
            elif change == 'two-descriptions':
                sbom['relationships'].append(dict(sbom['relationships'][0]))
            elif change == 'wrong-document':
                sbom['SPDXID'] = 'untrusted-document'
                sbom['relationships'][0]['spdxElementId'] = 'untrusted-document'
            elif change == 'duplicate-id':
                sbom['packages'].append(dict(sbom['packages'][0]))
            elif change == 'multiple-purls':
                sbom['packages'][0]['externalRefs'].append({'referenceLocator': 'pkg:oci/other@sha256%3A456'})
            elif change == 'wrong-purpose':
                sbom['packages'][0]['primaryPackagePurpose'] = 'LIBRARY'
            else:
                sbom['packages'][0]['externalRefs'][0]['referenceLocator'] = 'pkg:oci/unexpected/lfg@sha256%3A123'
            with self.subTest(change=change):
                self.assertTrue(any(kind == 'oci' for kind, _ in self.kinds(sbom)))


if __name__ == '__main__':
    unittest.main()
