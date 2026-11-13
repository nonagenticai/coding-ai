"""Security Insights Generator.

Generates security-related insights including vulnerability detection,
security hotspots, and common vulnerability patterns.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    BaseInsightGenerator,
    CodeInsight,
    InsightConfidence,
    InsightPriority,
    InsightType,
)

logger = logging.getLogger(__name__)


class SecurityInsightGenerator(BaseInsightGenerator):
    """Generates security-related insights."""

    async def generate(self, data: dict[str, Any]) -> list[CodeInsight]:
        """Generate security-related insights.

        Args:
            data: Analysis data containing security_analysis and other metrics.

        Returns:
            List of security-related CodeInsight objects.
        """
        insights: list[CodeInsight] = []

        security_analysis = data.get("security_analysis")
        if not security_analysis:
            return insights

        # Critical security issues
        critical_vulnerabilities = security_analysis.vulnerabilities_by_severity.get("critical", [])
        if critical_vulnerabilities:
            insights.append(
                CodeInsight(
                    id="security_critical",
                    insight_type=InsightType.SECURITY,
                    priority=InsightPriority.CRITICAL,
                    confidence=InsightConfidence.VERY_HIGH,
                    title="Critical Security Vulnerabilities Detected",
                    description=(
                        f"Found {len(critical_vulnerabilities)} critical security "
                        "vulnerabilities requiring immediate attention"
                    ),
                    rationale=(
                        "Critical vulnerabilities pose immediate security risks to the application"
                    ),
                    files=list(set(vuln.file_path for vuln in critical_vulnerabilities)),
                    recommendations=[
                        "Address critical vulnerabilities immediately",
                        "Implement security code review process",
                        "Add automated security scanning to CI/CD",
                        "Provide security training for development team",
                    ],
                    mitigation_suggestions=[
                        vuln.remediation_advice for vuln in critical_vulnerabilities[:3]
                    ],
                    estimated_effort="high",
                    evidence={
                        "vulnerability_count": len(critical_vulnerabilities),
                        "types": list(
                            set(vuln.vulnerability_type for vuln in critical_vulnerabilities)
                        ),
                    },
                    tags=["security", "critical", "vulnerability"],
                )
            )

        # Security hotspots
        if security_analysis.security_hotspots:
            insights.append(
                CodeInsight(
                    id="security_hotspots",
                    insight_type=InsightType.SECURITY,
                    priority=InsightPriority.HIGH,
                    confidence=InsightConfidence.HIGH,
                    title="Security Hotspots Identified",
                    description=(
                        f"Found {len(security_analysis.security_hotspots)} files with multiple security issues"
                    ),
                    rationale=(
                        "Files with multiple vulnerabilities indicate systemic security problems"
                    ),
                    files=security_analysis.security_hotspots,
                    recommendations=[
                        "Prioritize security review of hotspot files",
                        "Refactor security-critical code with proper validation",
                        "Implement defense-in-depth security patterns",
                        "Add comprehensive security testing",
                    ],
                    estimated_effort="high",
                    evidence={"hotspot_files": security_analysis.security_hotspots},
                    tags=["security", "hotspot", "review"],
                )
            )

        # Most common vulnerability insights
        if security_analysis.most_common_vulnerability:
            vuln_type = security_analysis.most_common_vulnerability
            vuln_count = len(security_analysis.vulnerabilities_by_type.get(vuln_type, []))

            insights.append(
                CodeInsight(
                    id=f"security_pattern_{vuln_type}",
                    insight_type=InsightType.SECURITY,
                    priority=InsightPriority.MEDIUM,
                    confidence=InsightConfidence.HIGH,
                    title=(f"Common Security Pattern: {vuln_type.replace('_', ' ').title()}"),
                    description=(
                        f"The most common vulnerability type is {vuln_type} with {vuln_count} instances"
                    ),
                    rationale=(
                        "Recurring vulnerability patterns indicate need for systematic remediation"
                    ),
                    recommendations=[
                        f"Implement coding standards to prevent {vuln_type}",
                        "Add automated detection for this vulnerability pattern",
                        "Provide targeted training on secure coding practices",
                        "Review and update security guidelines",
                    ],
                    estimated_effort="medium",
                    evidence={
                        "vulnerability_type": vuln_type,
                        "count": vuln_count,
                    },
                    tags=["security", "pattern", vuln_type],
                )
            )

        return insights
