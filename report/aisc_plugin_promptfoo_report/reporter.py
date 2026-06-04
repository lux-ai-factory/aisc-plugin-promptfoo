from sqlalchemy.orm import Session

from vera_report_plugin_interface.base_report_plugin import BaseReporterPlugin

from .data_loader import PromptfooDataLoader


class PROMPTFOOReportPlugin(BaseReporterPlugin):
    tool_name = "PROMPTFOO"

    def __init__(self, db_session: Session | None = None):
        super().__init__(db_session)

    def get_template_relative_path(self) -> str:
        return "templates/template_section.html.j2"

    def build_template_context(self, **kwargs: object) -> dict[str, object]:
        loader = PromptfooDataLoader(self.session)
        statistics = loader.compute_statistics()
        return {
            "tool_name": self.tool_name,
            "statistics": statistics,
            "tool_plots": kwargs.get("tool_plots") or {},
        }
