from cdm.event.common.ContractDetails import ContractDetails
from cdm.legaldocumentation.master.MasterAgreementTypeEnum import MasterAgreementTypeEnum
from cdm.legaldocumentation.master.MasterConfirmationTypeEnum import MasterConfirmationTypeEnum
from cdm.legaldocumentation.common.ContractualDefinitionsEnum import ContractualDefinitionsEnum
from cdm.legaldocumentation.common.MatrixTypeEnum import MatrixTypeEnum
from cdm.legaldocumentation.common.MatrixTermEnum import MatrixTermEnum
from cdm.legaldocumentation.common.LegalAgreementTypeEnum import LegalAgreementTypeEnum
from cdm.legaldocumentation.common.LegalAgreementIdentification import LegalAgreementIdentification
from cdm.legaldocumentation.common.ContractualMatrix import ContractualMatrix
from cdm.legaldocumentation.common.AgreementName import AgreementName
from cdm.legaldocumentation.common.LegalAgreement import LegalAgreement
from cdm.legaldocumentation.common.ContractualSupplementTypeEnum import ContractualSupplementTypeEnum
from cdm.legaldocumentation.common.ContractualTermsSupplement import ContractualTermsSupplement
from src.query.ragSearch import RAGSearch
from src.store.pgVector.multiModalPGVectorStore import MultimodalPGVectorStore
from src.embedding.multiModalEmbeddingWrapper import MultimodalEmbeddingWrapper
from cdm.legaldocumentation.master.MasterConfirmationAnnexTypeEnum import MasterConfirmationAnnexTypeEnum

class ContractDetailsBuilder:
    def __init__(self, connection_string, collection_name):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.rag_search = self.initialize_query_processor()

    def initialize_query_processor(self):
        embedding_wrapper = MultimodalEmbeddingWrapper(chunk_size=1000)
        pg_vector_store = MultimodalPGVectorStore(
            connection_string=self.connection_string,
            embedding_wrapper=embedding_wrapper,
            collection_name=self.collection_name
        )
        return RAGSearch(pg_vector_store)


    def is_enum_value(self, enum_class, value):
        try:
            enum_class(value)
            return True
        except ValueError:
            return False

    def buildAgreementNameMasterAgreement(self):
        kwargs = {}
        masterAgreementTypeEnum = self.rag_search.search_enum_in_documents(MasterAgreementTypeEnum, is_list=False)
        if(masterAgreementTypeEnum is not None):
            kwargs["masterAgreementType"] = masterAgreementTypeEnum
            kwargs["agreementType"] = LegalAgreementTypeEnum.MASTER_AGREEMENT
            # Create the AgreementName object
            agreement_name_instance = AgreementName(**kwargs)
            return agreement_name_instance
        else:
            return None

    def buildAgreementNameMasterConfirmation(self):
        kwargs = {}
        masterConfirmationTypeEnum = self.rag_search.search_enum_in_documents(MasterConfirmationTypeEnum, is_list=False)
        if(masterConfirmationTypeEnum is not None):
            kwargs["masterConfirmationType"] = masterConfirmationTypeEnum

        masterConfirmationAnnexTypeEnum = self.rag_search.search_enum_in_documents(MasterConfirmationAnnexTypeEnum, is_list=False)

        if(masterConfirmationAnnexTypeEnum is not None):
            kwargs["masterConfirmationAnnexType"] = masterConfirmationAnnexTypeEnum

        if(masterConfirmationTypeEnum is None):
            return None

        kwargs["agreementType"] = LegalAgreementTypeEnum.MASTER_CONFIRMATION
        # Create the AgreementName object
        agreement_name_instance = AgreementName(**kwargs)
        return agreement_name_instance

    def buildAgreementNameConfirmation(self):
        kwargs = {}
        contractualDefinitionsEnum = self.rag_search.search_enum_in_documents(ContractualDefinitionsEnum, is_list=True)
        contractualSupplementTypeEnum = self.rag_search.search_enum_in_documents(ContractualSupplementTypeEnum, is_list=True)

        matrixTypeEnum = self.rag_search.search_enum_in_documents(MatrixTypeEnum, is_list=False)
        matrixTermEnum  = self.rag_search.search_enum_in_documents(MatrixTermEnum, is_list=False)

        if(contractualDefinitionsEnum is not None):
            kwargs["contractualDefinitionsType"] = contractualDefinitionsEnum
        if(contractualSupplementTypeEnum):
            termsSupplement= []
            for term in contractualSupplementTypeEnum:
                termsSupplement.append(ContractualTermsSupplement(**{"contractualTermsSupplementType": term}))
            kwargs["contractualTermsSupplement"] = termsSupplement

        kwargs1 ={}
        if(matrixTypeEnum is not None):
            kwargs1["matrixType"] = matrixTypeEnum
        if(matrixTermEnum is not None):
            kwargs1["matrixTerm"] = matrixTermEnum
        if(matrixTypeEnum is not None):
            kwargs["contractualMatrix"] = [ContractualMatrix(**kwargs1)]

        if(kwargs.keys() is None):
            return None

        kwargs["agreementType"] = LegalAgreementTypeEnum.CONFIRMATION
        # Create the AgreementName object
        agreement_name_instance = AgreementName(**kwargs)
        return agreement_name_instance

    def buildDocumentation(self, agreement):
        legal_agreement_identification = LegalAgreementIdentification(**{"agreementName": agreement})
        return LegalAgreement(**{"legalAgreementIdentification": legal_agreement_identification})

    def buildContractDetails(self):
        agreement_names = []
        master_agreement = self.buildAgreementNameMasterAgreement()
        master_confirmation = self.buildAgreementNameMasterConfirmation()
        confirmation = self.buildAgreementNameConfirmation()

        if master_agreement:
            agreement_names.append(master_agreement)
        if master_confirmation:
            agreement_names.append(master_confirmation)
        if confirmation:
            agreement_names.append(confirmation)

        documents = [self.buildDocumentation(agreement) for agreement in agreement_names]
        contract_details = ContractDetails(**{"documentation": documents})
        return contract_details.model_dump_json(indent=4, exclude_none=True)