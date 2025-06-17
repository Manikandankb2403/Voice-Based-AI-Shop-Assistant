import pymongo
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QTextEdit, QFrame, QSizePolicy, QCheckBox)
from PyQt6.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
import threading
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from symspellpy import SymSpell, Verbosity
from fuzzywuzzy import process
from operator import itemgetter
from pymongo import MongoClient
from datetime import datetime
import json
import speech_recognition as sr
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from peft import PeftModel

class ChatWorker(QThread):
    response_signal = pyqtSignal(dict)
    
    def __init__(self, chatbot, message, input_method):
        super().__init__()
        self.chatbot = chatbot
        self.message = message
        self.input_method = input_method
        self._running = True
    
    def run(self):
        if self._running:
            response = self.chatbot.process_message(self.message, self.input_method)
            self.response_signal.emit(response)
    
    def stop(self):
        self._running = False
        self.quit()
        self.wait()

class VaseegrahVedaChatbot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vaseegrah Veda - AI Shop Assistant")
        self.setGeometry(100, 100, 900, 700)
        
        # Customer info
        self.customer_id = None
        self.phone_number = None
        
        # Session state
        self.session = {
            "cart": [],
            "name": "",
            "address": "",
            "phone_number": "",
            "paymentConfirmed": False,
            "total_price": 0,
            "input_method": "text"
        }

        # Waiting flags for multi-turn interactions
        self.waiting_for_name = False
        self.waiting_for_address = False
        self.waiting_for_payment_method = False
        self.waiting_for_new_name = False
        self.waiting_for_new_address = False

        # Input mode toggle (mic-only or text-only)
        self.mic_only_mode = False

        # List to track active threads
        self.active_threads = []

        # Products list
        self.products = []

        # Speech recognizer and custom Whisper model
        self.recognizer = sr.Recognizer()
        try:
            self.processor = WhisperProcessor.from_pretrained("./good/")
            self.model = WhisperForConditionalGeneration.from_pretrained("./model_whisper/")
            self.model = PeftModel.from_pretrained(self.model, "./whisper-finetuned-final/")
            self.model.eval()
            self.model.generation_config.forced_decoder_ids = None
            print("Custom Whisper model and processor loaded successfully")
        except Exception as e:
            error_msg = f"Error loading custom Whisper model/processor: {e}"
            print(error_msg)
            self.show_error(error_msg)
            self.processor = None
            self.model = None

        # Setup components
        self.setup_database()
        self.setup_products_from_db()
        self.setup_text_processor()
        self.setup_product_names()
        self.setup_vector_store()
        self.setup_llm()
        self.setup_prompts()
        self.setup_chain()
        
        # Initialize UI
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.show_login_screen()
        
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#ECEFF1"))
        self.setPalette(palette)

    def show_error(self, message):
        error_html = f'<div style="color: #ff3333; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">{message}</div>'
        if hasattr(self, 'chat_display'):
            self.chat_display.append(error_html)
            self.chat_display.ensureCursorVisible()
        print(message)

    def setup_database(self):
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["model_rag"]
            self.customers_collection = self.db["customers"]
            self.carts_collection = self.db["carts"]
            self.orders_collection = self.db["orders"]
            self.products_collection = self.db["products"]
            self.db_status = True
            print("Database connected successfully")
        except Exception as e:
            self.db_status = False
            error_msg = f"Database connection error: {e}"
            self.show_error(error_msg)

    def setup_products_from_db(self):
        try:
            self.products = list(self.products_collection.find({}, {"_id": 0}))
            for product in self.products:
                if not all(key in product for key in ["name", "price", "stock"]):
                    print(f"Invalid product document: {product}")
                    self.show_error(f"Product '{product.get('name', 'Unknown')}' missing required fields")
            print(f"Loaded {len(self.products)} products from database")
        except Exception as e:
            self.show_error(f"Error fetching products from database: {e}")
            self.products = []

    def setup_text_processor(self):
        try:
            self.sym_spell = SymSpell()
            dictionary_path = "../encoded_files/frequency_dictionary_en_82_765.txt"
            if os.path.exists(dictionary_path):
                self.sym_spell.load_dictionary(dictionary_path, 0, 1)
                print("Spell checker dictionary loaded successfully")
            else:
                warning_msg = f"Warning: Dictionary file '{dictionary_path}' not found. Spell correction disabled."
                print(warning_msg)
                self.show_error(warning_msg)
        except Exception as e:
            error_msg = f"Spell checker initialization failed: {e}"
            self.show_error(error_msg)

    def setup_product_names(self):
        try:
            self.product_names = [product["name"] for product in self.products]
            print(f"Loaded {len(self.product_names)} product names from database")
        except Exception as e:
            self.product_names = []
            error_msg = f"Product names loading failed: {e}"
            self.show_error(error_msg)

    class QAPairLoader(TextLoader):
        def load(self):
            docs = super().load()
            processed = []
            qa_pattern = re.compile(r'\"(.*?)\",\"(.*?)\"(?=\n|$)', re.DOTALL)
            for doc in docs:
                matches = qa_pattern.findall(doc.page_content)
                for question, answer in matches:
                    processed.append(f"Q: {question.strip()}\nA: {answer.strip()}")
            return [doc.__class__(page_content="\n\n".join(processed), metadata=doc.metadata)]

    def setup_vector_store(self):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\nQ:", "\nQ:"],
                keep_separator=True
            )
            documents = []
            for file_path in ["../encoded_files/rag_s1.txt", "../encoded_files/Vaseegrah veda rag file.txt"]:
                try:
                    loader = self.QAPairLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    documents.extend(split_docs)
                    print(f"Loaded {len(split_docs)} documents from {file_path}")
                except Exception as e:
                    error_msg = f"Error loading {file_path}: {e}"
                    self.show_error(error_msg)
            embeddings = OllamaEmbeddings(model="gemma2:2b")
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "lambda_mult": 0.75, "score_threshold": 0.7}
            )
            self.rag_status = True
            print("Vector store initialized successfully")
        except Exception as e:
            self.rag_status = False
            error_msg = f"Vector store initialization failed: {e}"
            self.show_error(error_msg)

    def setup_llm(self):
        try:
            self.llm = ChatOllama(
                model="gemma2:2b",
                temperature=0.5,
                system="You're Srija, Vaseegrah Veda's expert assistant. Prioritize document content and maintain friendly tone."
            )
            self.llm_status = True
            print("LLM initialized successfully")
        except Exception as e:
            self.llm_status = False
            error_msg = f"LLM initialization failed: {e}"
            self.show_error(error_msg)

    def setup_prompts(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "stock_info", "name"],
            template="""
            ROLE: Vaseegrah Veda Customer Support
            TASK: Answer using EXACT information from documents when available
            
            USER NAME: {name} (if provided, address the user by name in the response)
            
            DOCUMENT CONTEXT:
            {context}
            
            CURRENT STOCK:
            {stock_info}
            
            USER QUESTION: {question}
            
            RESPONSE RULES:
            1. If USER NAME is provided, start the response by addressing the user by name
            2. Start with an emoji relevant to the question
            3. Use direct quotes from context for founder/owner questions
            4. For products: Combine document info with stock data
            5. If unsure: "Let me check that for you..."
            6. Max 3 emojis per response
            
            FINAL RESPONSE:
            """
        )
        
        self.cart_action_prompt = PromptTemplate(
            input_variables=["action", "product", "quantity", "name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. Generate a friendly response for {action} {quantity} {product}(s) to/from the cart for {name}.
            Keep it casual and use emojis.
            """
        )
        
        self.view_cart_prompt = PromptTemplate(
            input_variables=["cart_summary", "name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. Here's the cart summary for {name}: {cart_summary}. Generate a friendly response to show them their cart.
            """
        )
        
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the intent of the following query and extract all products and quantities if applicable.
            
            Possible intents:
            - view_cart: User wants to see their cart contents.
            - checkout: User wants to proceed to checkout.
            - add_to_cart: User wants to add products to their cart. Extract all product names and quantities.
            - remove_from_cart: User wants to remove products from their cart. Extract all product names and quantities.
            - update_name: User wants to update their name (e.g., "I need to update my name", "note my name").
            - update_address: User wants to update their address (e.g., "I need to update my address", "note my address").
            - general_query: Any other query that doesn't fit the above intents.
            
            Examples:
            - "show my cart" -> intent: view_cart
            - "I want to checkout" -> intent: checkout
            - "add 2 rose hydrosol and 1 lavender soap" -> intent: add_to_cart
              [product: rose hydrosol, quantity: 2]
              [product: lavender soap, quantity: 1]
            - "buy three lavender soaps and two rose hydrosols" -> intent: add_to_cart
              [product: lavender soap, quantity: 3]
              [product: rose hydrosol, quantity: 2]
            - "remove one lavender soap and 2 rose hydrosols" -> intent: remove_from_cart
              [product: lavender soap, quantity: 1]
              [product: rose hydrosol, quantity: 2]
            - "I need to update my address" -> intent: update_address
            - "note my address as Chennai" -> intent: update_address
            - "I need to update my name to Priya" -> intent: update_name
            - "note my name as Priya" -> intent: update_name
            - "tell me about your products" -> intent: general_query
            - "i want rosemary hydrosol" -> intent: add_to_cart
              [product: rosemary hydrosol, quantity: 1]
            
            Query: "{query}"
            
            Provide the intent and, if applicable, all product-quantity pairs or extracted name/address in the following format:
            intent: <intent>
            [product: <product>, quantity: <quantity>] (for add_to_cart/remove_from_cart)
            [name: <name>] (for update_name)
            [address: <address>] (for update_address)
            """
        )
        
        self.stock_issue_prompt = PromptTemplate(
            input_variables=["product", "available", "name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. {name}, you tried to add {product} to your cart, 
            but there are only {available} left. Please adjust the quantity or choose another product. 😊
            """
        )

        self.out_of_stock_prompt = PromptTemplate(
            input_variables=["product", "name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. Sorry, {name}, {product} is currently out of stock. 😔 
            Check back later or explore similar products! 🌿
            """
        )

        self.insufficient_stock_checkout_prompt = PromptTemplate(
            input_variables=["items_list", "name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. {name}, you're trying to checkout, but these items have insufficient stock:\n{items_list}\nPlease adjust your cart by reducing quantities or removing items. 😊
            """
        )

        self.update_name_prompt = PromptTemplate(
            input_variables=["name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. Got it, {name}! Please tell me your new name. 😊
            """
        )

        self.update_address_prompt = PromptTemplate(
            input_variables=["name"],
            template="""
            You're Srija, Vaseegrah Veda's assistant. Alright, {name}! Please share your new address. 📍
            """
        )

        print("Prompt templates initialized")

    def setup_chain(self):
        self.chain = RunnableParallel(
            context=itemgetter("question") | self.retriever | self.format_docs,
            question=itemgetter("question"),
            stock_info=lambda x: self.get_stock_info(),
            name=lambda x: self.session.get("name", "")
        ) | self.prompt_template | self.llm
        print("Chain setup completed")

    def format_docs(self, docs):
        boosted = []
        for doc in docs:
            content = doc.page_content
            if "vijaya mahadevan" in content.lower():
                content = "🌟 FOUNDER INFO 🌟\n" + content
            boosted.append(f"📄 {content}")
        return "\n\n".join(boosted) or "No relevant information found"

    def correct_text(self, text):
        try:
            words = text.split()
            return " ".join([
                self.sym_spell.lookup(word, Verbosity.CLOSEST, 2)[0].term
                if self.sym_spell.lookup(word, Verbosity.CLOSEST, 2)
                else word
                for word in words
            ])
        except Exception:
            return text

    def match_product(self, user_query):
        if not self.product_names:
            error_msg = "No product names loaded from database. Cannot match products."
            self.show_error(error_msg)
            return None
        try:
            result = process.extractOne(user_query, self.product_names, score_cutoff=80)
            if result:
                matched_name, score = result
                print(f"Product match: '{user_query}' -> '{matched_name}' with score {score}")
                return matched_name
            else:
                print(f"No product match found for: '{user_query}' with score >= 80")
                return None
        except Exception as e:
            error_msg = f"Error in product matching: {e}"
            self.show_error(error_msg)
            return None

    def get_stock_info(self):
        try:
            stock_list = [f"- {p['name']}: ₹{p['price']} ({p['stock'] if p['stock'] is not None else 'Not managed'} available)" 
                          for p in self.products]
            return "\n".join(stock_list) or "No products currently in stock"
        except Exception as e:
            error_msg = f"Error retrieving stock info from database: {e}"
            self.show_error(error_msg)
            return "⚠️ Could not retrieve stock information"

    def generate_response(self, query):
        try:
            if not query.strip():
                return {"message": f"😊 Hi there, {self.session.get('name', '')}! How can I assist you today?", "session": self.session.copy()}
            cleaned = re.sub(r'[^\w\s]', '', query).lower()
            corrected = self.correct_text(cleaned)
            print(f"Corrected query: '{query}' -> '{corrected}'")
            if any(kw in corrected for kw in ["founder", "owner", "vijaya"]):
                response = self.chain.invoke({"question": corrected})
                return {"message": response.content, "session": self.session.copy()}
            product_match = self.match_product(corrected)
            if product_match:
                query_with_product = f"Tell me about {product_match}. User query: {corrected}"
                response = self.chain.invoke({"question": query_with_product})
                return {"message": response.content, "session": self.session.copy()}
            response = self.chain.invoke({"question": corrected})
            return {"message": response.content, "session": self.session.copy()}
        except Exception as e:
            error_msg = f"😅 Let me check that again... Could you please rephrase your question? (Error: {e})"
            self.show_error(f"Error generating response: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def get_product_price(self, product_name):
        try:
            product = next((p for p in self.products if p["name"] == product_name), None)
            price = float(product["price"]) if product and product["price"] else None
            print(f"Product price for '{product_name}': {price}")
            return price
        except Exception as e:
            error_msg = f"Error fetching price for '{product_name}' from database: {e}"
            self.show_error(error_msg)
            return None

    def add_to_cart(self, product_name, quantity):
        if not self.customer_id:
            return {"message": "Please log in first to add items to your cart.", "session": self.session.copy()}
        if quantity <= 0:
            return {"message": "Please specify a positive quantity to add. 😊", "session": self.session.copy()}
        try:
            product = next((p for p in self.products if p["name"] == product_name), None)
            if not product:
                return {"message": f"Sorry, '{product_name}' isn't available in our inventory. 😔", "session": self.session.copy()}
            stock = product["stock"] if product["stock"] is not None else float('inf')
            if stock == 0:
                response = self.llm.invoke(self.out_of_stock_prompt.format(product=product_name, name=self.session.get("name", "")))
                return {"message": response.content, "session": self.session.copy()}
            if quantity > stock:
                response = self.llm.invoke(self.stock_issue_prompt.format(product=product_name, available=stock, name=self.session.get("name", "")))
                return {"message": response.content, "session": self.session.copy()}
            price = float(product["price"]) if product["price"] else 0
            cart = self.carts_collection.find_one({"customer_id": self.customer_id}) or {
                "customer_id": self.customer_id, "items": [], "total_price": 0}
            found = False
            for item in cart["items"]:
                if item["product_name"] == product_name:
                    item["quantity"] += quantity
                    found = True
                    break
            if not found:
                cart["items"].append({"product_name": product_name, "quantity": quantity, "price": price})
            cart["total_price"] = sum(item["quantity"] * item["price"] for item in cart["items"])
            self.carts_collection.update_one(
                {"customer_id": self.customer_id},
                {"$set": cart},
                upsert=True
            )
            self.session["cart"] = cart["items"]
            self.session["total_price"] = cart["total_price"]
            response = self.llm.invoke(self.cart_action_prompt.format(action="added", product=product_name, quantity=quantity, name=self.session.get("name", "")))
            return {"message": response.content, "session": self.session.copy()}
        except Exception as e:
            error_msg = f"There was an error adding '{product_name}' to your cart: {e} 😅"
            self.show_error(error_msg)
            return {"message": error_msg, "session": self.session.copy()}

    def remove_from_cart(self, product_name, quantity):
        try:
            cart = self.carts_collection.find_one({"customer_id": self.customer_id})
            if not cart or not cart["items"]:
                return {"message": "Your cart's empty!", "session": self.session.copy()}
            for item in cart["items"]:
                if item["product_name"] == product_name:
                    if item["quantity"] > quantity:
                        item["quantity"] -= quantity
                    else:
                        cart["items"].remove(item)
                    cart["total_price"] = sum(item["quantity"] * item["price"] for item in cart["items"])
                    self.carts_collection.update_one(
                        {"customer_id": self.customer_id},
                        {"$set": cart}
                    )
                    self.session["cart"] = cart["items"]
                    self.session["total_price"] = cart["total_price"]
                    response = self.llm.invoke(self.cart_action_prompt.format(action="removed", product=product_name, quantity=quantity, name=self.session.get("name", "")))
                    return {"message": response.content, "session": self.session.copy()}
            return {"message": f"'{product_name}' isn't in your cart.", "session": self.session.copy()}
        except Exception as e:
            error_msg = f"There was an error removing '{product_name}' from your cart: {e} 😅"
            self.show_error(f"Error removing from cart: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def view_cart(self):
        try:
            cart = self.carts_collection.find_one({"customer_id": self.customer_id})
            if cart and cart["items"]:
                cart_summary = "\n".join([f"- {item['product_name']}: {item['quantity']} @ ₹{item['price']}" for item in cart["items"]])
                total_price = cart["total_price"]
                summary = f"{cart_summary}\nTotal: ₹{total_price}"
                self.session["cart"] = cart["items"]
                self.session["total_price"] = total_price
                response = self.llm.invoke(self.view_cart_prompt.format(cart_summary=summary, name=self.session.get("name", "")))
                return {"message": response.content, "session": self.session.copy()}
            else:
                self.session["total_price"] = 0
                return {"message": "Your cart's empty right now.", "session": self.session.copy()}
        except Exception as e:
            error_msg = f"There was an error viewing your cart: {e} 😅"
            self.show_error(f"Error viewing cart: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def update_customer_name(self, name):
        try:
            name = name.strip()
            if not name:
                print("Name is empty, not updating.")
                return False
            self.customers_collection.update_one(
                {"_id": self.customer_id},
                {"$set": {"name": name}}
            )
            self.session["name"] = name
            print(f"Updated name for customer {self.customer_id}: {name}")
            return True
        except Exception as e:
            error_msg = f"Error updating name: {e}"
            self.show_error(error_msg)
            return False

    def update_customer_address(self, address):
        try:
            address = address.strip()
            if not address:
                print("Address is empty, not updating.")
                return False
            self.customers_collection.update_one(
                {"_id": self.customer_id},
                {"$set": {"address": address}}
            )
            self.session["address"] = address
            print(f"Updated address for customer {self.customer_id}: {address}")
            return True
        except Exception as e:
            error_msg = f"Error updating address: {e}"
            self.show_error(error_msg)
            return False

    def checkout(self):
        try:
            cart = self.carts_collection.find_one({"customer_id": self.customer_id})
            if not cart or not cart["items"]:
                return {"message": "Your cart is empty! Add some products first. 😊", "session": self.session.copy()}
            customer = self.customers_collection.find_one({"_id": self.customer_id})
            if not customer:
                return {"message": "I couldn't find your details. Could you please log in again?", "session": self.session.copy()}
            name = customer.get("name", "").strip()
            address = customer.get("address", "").strip()
            if not name:
                self.waiting_for_name = True
                return {"message": "Hey there! Before we proceed, could you please tell me your name? 😊", "session": self.session.copy()}
            elif not address:
                self.waiting_for_address = True
                return {"message": f"Hi {name}! I just need your address to deliver your order. Could you share it with me? 📍", "session": self.session.copy()}
            else:
                self.waiting_for_payment_method = True
                return {"message": f"All set, {name}! How would you like to pay? Choose one: UPI, Card, or Net Banking. 💳", "session": self.session.copy()}
        except Exception as e:
            error_msg = f"Oops! There was an error while checking out: {e} 😅"
            self.show_error(f"Error during checkout: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def process_payment(self, payment_method):
        try:
            cart = self.carts_collection.find_one({"customer_id": self.customer_id})
            if not cart or not cart["items"]:
                return {"message": "Your cart is empty! Add something first. 😊", "session": self.session.copy()}
            insufficient_stock_items = []
            for item in cart["items"]:
                product_name = item["product_name"]
                quantity_ordered = item["quantity"]
                product = next((p for p in self.products if p["name"] == product_name), None)
                if not product:
                    insufficient_stock_items.append(f"{product_name}: not available")
                elif product["stock"] is not None and product["stock"] < quantity_ordered:
                    insufficient_stock_items.append(f"{product_name}: only {product['stock']} left")
            if insufficient_stock_items:
                items_list = "\n".join(insufficient_stock_items)
                response = self.llm.invoke(self.insufficient_stock_checkout_prompt.format(items_list=items_list, name=self.session.get("name", "")))
                return {"message": response.content, "session": self.session.copy()}
            customer = self.customers_collection.find_one({"_id": self.customer_id})
            name = customer.get("name", "Customer")
            order = {
                "customer_id": self.customer_id,
                "items": cart["items"],
                "total_price": cart["total_price"],
                "payment_method": payment_method.capitalize(),
                "status": "pending",
                "created_at": datetime.now()
            }
            order_id = self.orders_collection.insert_one(order).inserted_id
            print(f"Created order ID: {order_id}")
            self.carts_collection.delete_one({"customer_id": self.customer_id})
            self.session["cart"] = []
            self.session["total_price"] = 0
            self.session["paymentConfirmed"] = True
            return {"message": f"✅ Order placed successfully, {name}! Your order ID is **{order_id}**. Thanks for choosing {payment_method.capitalize()}! 🛍️", "session": self.session.copy()}
        except Exception as e:
            error_msg = f"Oops! There was an error processing your payment: {e} 😅"
            self.show_error(f"Error processing payment: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def process_message(self, query, input_method="text"):
        print(f"\nProcessing message: '{query}' (Input method: {input_method})")
        self.session["input_method"] = input_method
        if not query.strip():
            return {"message": f"😊 Hi there, {self.session.get('name', '')}! How can I assist you today?", "session": self.session.copy()}
        if query.lower() in ["exit", "quit"]:
            self.cleanup_threads()
            QApplication.quit()
            return {"message": "Goodbye! 🌸", "session": self.session.copy()}
        
        # Handle multi-turn interactions
        if self.waiting_for_name:
            name_patterns = [
                r"(?:my name is|note my name as|i am|name is)\s+(.+)",
                r"(.+)"
            ]
            name = query.strip()
            for pattern in name_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    name = match.group(1).strip()
                    break
            if self.update_customer_name(name):
                self.waiting_for_name = False
                customer = self.customers_collection.find_one({"_id": self.customer_id})
                if not customer.get("address", "").strip():
                    self.waiting_for_address = True
                    return {"message": f"Thanks, {name}! Now, could you please share your address for delivery? 📍", "session": self.session.copy()}
                else:
                    self.waiting_for_payment_method = True
                    return {"message": f"Got it, {name}! How would you like to pay? Choose one: UPI, Card, or Net Banking. 💳", "session": self.session.copy()}
            else:
                return {"message": "Sorry, that name doesn't seem valid. Please provide a valid name. 😊", "session": self.session.copy()}
        elif self.waiting_for_address:
            address_patterns = [
                r"(?:i am from|my address is|deliver to)\s+(.+)",
                r"(.+)"
            ]
            address = query.strip()
            for pattern in address_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    address = match.group(1).strip()
                    break
            if self.update_customer_address(address):
                self.waiting_for_address = False
                customer = self.customers_collection.find_one({"_id": self.customer_id})
                name = customer.get("name", "Customer")
                self.waiting_for_payment_method = True
                return {"message": f"Awesome, {name}! I've saved your address as '{address}'. How would you like to pay? Choose one: UPI, Card, or Net Banking. 💳", "session": self.session.copy()}
            else:
                return {"message": "Sorry, that address doesn't seem valid. Please provide a valid address. 📍", "session": self.session.copy()}
        elif self.waiting_for_new_name:
            name_patterns = [
                r"(?:my name is|note my name as|i am|name is)\s+(.+)",
                r"(.+)"
            ]
            name = query.strip()
            for pattern in name_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    name = match.group(1).strip()
                    break
            if self.update_customer_name(name):
                self.waiting_for_new_name = False
                return {"message": f"Updated your name to '{name}', {name}! Anything else I can help with? 😊", "session": self.session.copy()}
            else:
                return {"message": "Sorry, that name doesn't seem valid. Please provide a valid name. 😊", "session": self.session.copy()}
        elif self.waiting_for_new_address:
            address_patterns = [
                r"(?:i am from|my address is|deliver to)\s+(.+)",
                r"(.+)"
            ]
            address = query.strip()
            for pattern in address_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    address = match.group(1).strip()
                    break
            if self.update_customer_address(address):
                self.waiting_for_new_address = False
                name = self.session.get("name", "Customer")
                return {"message": f"Updated your address to '{address}', {name}! Anything else I can help with? 📍", "session": self.session.copy()}
            else:
                return {"message": "Sorry, that address doesn't seem valid. Please provide a valid address. 📍", "session": self.session.copy()}
        elif self.waiting_for_payment_method:
            payment_method = query.strip().lower()
            valid_methods = ["upi", "card", "net banking"]
            pattern = r"(?:pay\s*(?:with|via|by|using)?\s*)?(upi|card|net\s*banking)"
            match = re.search(pattern, payment_method)
            detected_method = match.group(1) if match else None
            if detected_method:
                detected_method = "net banking" if "net" in detected_method else detected_method
                self.waiting_for_payment_method = False
                return self.process_payment(detected_method)
            else:
                return {"message": "Hmm, that’s not quite right. Please choose one: UPI, Card, or Net Banking. 💳", "session": self.session.copy()}

        try:
            full_prompt = self.intent_prompt.format(query=query)
            response = self.llm.invoke(full_prompt)
            print(f"Intent classification response: {response.content}")
            lines = response.content.strip().split("\n")
            intent = "general_query"
            products = []
            name = None
            address = None
            for line in lines:
                line = line.strip("[]")
                if line.startswith("intent:"):
                    intent = line.split(":", 1)[1].strip()
                elif line.startswith("product:"):
                    parts = line.split(", quantity:")
                    product = parts[0].split(":", 1)[1].strip()
                    quantity = parts[1].strip() if len(parts) > 1 else "1"
                    products.append((product, quantity))
                elif line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("address:"):
                    address = line.split(":", 1)[1].strip()
            print(f"Parsed intent: {intent}, products: {products}, name: {name}, address: {address}")
            if intent == "view_cart":
                return self.view_cart()
            elif intent == "checkout":
                return self.checkout()
            elif intent == "add_to_cart":
                if not products:
                    return {"message": "Which product would you like to add to your cart? 🤔", "session": self.session.copy()}
                messages = []
                for product, quantity in products:
                    matched_product = self.match_product(product)
                    if matched_product:
                        qty = int(quantity) if quantity.isdigit() else 1
                        result = self.add_to_cart(matched_product, qty)
                        messages.append(result["message"])
                    else:
                        messages.append(f"Sorry, I couldn’t find '{product}'. Would you like to see our product list? 📋")
                return {"message": "\n".join(messages), "session": self.session.copy()}
            elif intent == "remove_from_cart":
                if not products:
                    return {"message": "Which product would you like to remove from your cart? 🤔", "session": self.session.copy()}
                messages = []
                for product, quantity in products:
                    matched_product = self.match_product(product)
                    if matched_product:
                        qty = int(quantity) if quantity.isdigit() else 1
                        result = self.remove_from_cart(matched_product, qty)
                        messages.append(result["message"])
                    else:
                        messages.append(f"Sorry, I couldn’t find '{product}' in your cart. 😕")
                return {"message": "\n".join(messages), "session": self.session.copy()}
            elif intent == "update_name":
                if name:
                    if self.update_customer_name(name):
                        return {"message": f"Updated your name to '{name}', {name}! Anything else I can help with? 😊", "session": self.session.copy()}
                    else:
                        return {"message": "Sorry, that name doesn't seem valid. Please provide a valid name. 😊", "session": self.session.copy()}
                else:
                    self.waiting_for_new_name = True
                    response = self.llm.invoke(self.update_name_prompt.format(name=self.session.get("name", "")))
                    return {"message": response.content, "session": self.session.copy()}
            elif intent == "update_address":
                if address:
                    if self.update_customer_address(address):
                        name = self.session.get("name", "Customer")
                        return {"message": f"Updated your address to '{address}', {name}! Anything else I can help with? 📍", "session": self.session.copy()}
                    else:
                        return {"message": "Sorry, that address doesn't seem valid. Please provide a valid address. 📍", "session": self.session.copy()}
                else:
                    self.waiting_for_new_address = True
                    response = self.llm.invoke(self.update_address_prompt.format(name=self.session.get("name", "")))
                    return {"message": response.content, "session": self.session.copy()}
            else:
                info_keywords = ["about", "what is", "tell me", "describe", "information", "details", "uses", "benefits"]
                if not products and any(cmd in query.lower() for cmd in ["add", "buy", "purchase"]):
                    words = query.lower().split()
                    prod_start = words.index(next(cmd for cmd in ["add", "buy", "purchase"] if cmd in words)) + 1
                    if prod_start < len(words):
                        potential_product = " ".join(words[prod_start:])
                        matched_product = self.match_product(potential_product)
                        if matched_product:
                            return self.add_to_cart(matched_product, 1)
                if products and not any(keyword in query.lower() for keyword in info_keywords):
                    messages = []
                    for product, quantity in products:
                        matched_product = self.match_product(product)
                        if matched_product:
                            qty = int(quantity) if quantity.isdigit() else 1
                            result = self.add_to_cart(matched_product, qty)
                            messages.append(result["message"])
                        else:
                            messages.append(f"Sorry, I couldn’t find '{product}'. 😕")
                    return {"message": "\n".join(messages), "session": self.session.copy()}
                return self.generate_response(query)
        except Exception as e:
            error_msg = f"Oops! Something went wrong processing your message: {e} 😅"
            self.show_error(f"Error in process_message: {e}")
            return {"message": error_msg, "session": self.session.copy()}

    def transcribe_audio(self, audio_path):
        try:
            speech_array, sampling_rate = torchaudio.load(audio_path)
            if sampling_rate != 16000:
                transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                speech_array = transform(speech_array)
            input_features = self.processor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            error_msg = f"Error in transcribe_audio: {e}"
            self.show_error(error_msg)
            return None

    def record_audio(self):
        if not self.mic_only_mode:
            self.chat_display.append(f'<div style="color: #ff3333; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">Mic input is disabled. Toggle "Mic Only" on to use the microphone.</div>')
            self.chat_display.ensureCursorVisible()
            return
        if not self.model or not self.processor:
            error_msg = "Whisper model not loaded. Please check setup. 😅"
            self.show_error(error_msg)
            return
        try:
            print("Available microphones:", sr.Microphone.list_microphone_names())
            with sr.Microphone() as source:
                self.chat_display.append(f'<div style="color: #999999; font-style: italic; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">Listening... 🎤 (Say something within 10 seconds)</div>')
                self.chat_display.ensureCursorVisible()
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Energy threshold: {self.recognizer.energy_threshold}")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                print("Audio captured successfully!")
                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(audio.get_wav_data())
                print(f"Audio saved to {temp_audio_file}")
                audio_info = torchaudio.info(temp_audio_file)
                duration = audio_info.num_frames / audio_info.sample_rate
                print(f"Audio duration: {duration:.2f} seconds")
                transcription = self.transcribe_audio(temp_audio_file)
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
                    print("Temporary audio file removed")
                self.chat_display.append(f'<div style="color: #999999; font-style: italic; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;"></div>')
                if transcription and len(transcription.strip()) > 3:
                    print(f"Recognized speech (Whisper): '{transcription}'")
                    self.add_user_message(transcription)
                    worker = ChatWorker(self, transcription, "mic")
                    self.active_threads.append(worker)
                    worker.response_signal.connect(self.update_chat_with_response)
                    worker.start()
                else:
                    error_msg = "Sorry, I couldn’t understand that. Could you speak again or type it? 🎤"
                    self.show_error(error_msg)
        except sr.WaitTimeoutError:
            error_msg = "No speech detected. Try again! 🎤 (Ensure mic is on and speak clearly)"
            self.show_error(error_msg)
        except sr.UnknownValueError:
            error_msg = "Couldn’t understand the audio. Try speaking louder or clearer! 🎤"
            self.show_error(error_msg)
        except Exception as e:
            error_msg = f"Error recording audio: {e} 😅"
            self.show_error(error_msg)

    def show_login_screen(self):
        self.login_widget = QWidget()
        login_layout = QVBoxLayout(self.login_widget)
        login_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header = QLabel("🌿 Welcome to Vaseegrah Veda! 🌿")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gradient = QLinearGradient(0, 0, 900, 0)
        gradient.setColorAt(0, QColor("#26A69A"))
        gradient.setColorAt(1, QColor("#01579B"))
        header.setStyleSheet("color: white; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #26A69A, stop:1 #01579B); padding: 20px;")
        login_layout.addWidget(header)
        
        srija_label = QLabel("AI Shop Assistant")
        srija_label.setFont(QFont("Arial", 14))
        srija_label.setStyleSheet("color: #FF7043;")
        login_layout.addWidget(srija_label)
        
        phone_frame = QHBoxLayout()
        phone_label = QLabel("Enter your phone number:")
        phone_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        phone_label.setStyleSheet("color: #01579B;")
        phone_frame.addWidget(phone_label)
        
        self.phone_entry = QLineEdit()
        self.phone_entry.setFont(QFont("Arial", 12))
        self.phone_entry.setPlaceholderText("Enter 10-digit number")
        self.phone_entry.setStyleSheet("padding: 8px; border: 1px solid #26A69A; border-radius: 5px;")
        self.phone_entry.setFixedWidth(200)
        phone_frame.addWidget(self.phone_entry)
        login_layout.addLayout(phone_frame)
        
        login_button = QPushButton("Get Started")
        login_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        login_button.setStyleSheet("""
            QPushButton {background-color: #FF7043; color: white; padding: 10px; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #F4511E;}
        """)
        login_button.clicked.connect(self.handle_login)
        login_layout.addWidget(login_button)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 12))
        self.chat_display.setStyleSheet("""
            QTextEdit {background-color: #FFFFFF; border: none; padding: 15px; border-radius: 5px;}
        """)
        login_layout.addWidget(self.chat_display)
        
        status_frame = QWidget()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        status_frame.setStyleSheet("background-color: #01579B;")
        db_status = "✅" if self.db_status else "❌"
        rag_status = "✅" if self.rag_status else "❌"
        llm_status = "✅" if self.llm_status else "❌"
        status_label = QLabel(f"Database: {db_status} | RAG: {rag_status} | LLM: {llm_status}")
        status_label.setFont(QFont("Arial", 9))
        status_label.setStyleSheet("color: white;")
        status_layout.addWidget(status_label)
        login_layout.addWidget(status_frame, alignment=Qt.AlignmentFlag.AlignBottom)
        
        self.layout.addWidget(self.login_widget)

    def is_valid_indian_phone(self, phone_number):
        phone_number = re.sub(r'\D', '', phone_number)
        return bool(re.match(r'^[6789]\d{9}$', phone_number))

    def handle_login(self):
        phone_number = self.phone_entry.text().strip()
        if not phone_number or phone_number == "Enter 10-digit number":
            self.show_error("Please enter your phone number")
            return
        if not self.is_valid_indian_phone(phone_number):
            self.show_error("Invalid phone number! Enter a 10-digit Indian number starting with 6, 7, 8, or 9.")
            return
        try:
            customer = self.customers_collection.find_one({"phone_number": phone_number})
            print(f"Login attempt for phone: {phone_number}")
            print(f"Customer found: {customer}")
            if customer:
                self.customer_id = customer["_id"]
                self.session["name"] = customer.get("name", "")
                self.session["address"] = customer.get("address", "")
                self.session["phone_number"] = phone_number
                cart = self.carts_collection.find_one({"customer_id": self.customer_id})
                self.session["cart"] = cart["items"] if cart and cart.get("items") else []
                self.session["total_price"] = cart["total_price"] if cart and cart.get("total_price") else 0
                welcome_message = f"Welcome back, {self.session['name']}! 🌟" if self.session["name"] else "Welcome back! 🌟"
                self.show_chat_interface({"message": welcome_message, "session": self.session.copy()})
            else:
                result = self.customers_collection.insert_one({
                    "phone_number": phone_number,
                    "name": "",
                    "address": ""
                })
                self.customer_id = result.inserted_id
                self.session["name"] = ""
                self.session["address"] = ""
                self.session["phone_number"] = phone_number
                self.session["cart"] = []
                self.session["total_price"] = 0
                print(f"Created new customer with ID: {self.customer_id}")
                self.show_chat_interface({"message": "Welcome to Vaseegrah Veda! 🌿", "session": self.session.copy()})
        except Exception as e:
            error_msg = f"Failed to log in: {e}"
            self.show_error(error_msg)

    def toggle_input_mode(self, state):
        self.mic_only_mode = state
        if self.mic_only_mode:
            self.message_entry.setEnabled(False)
            self.send_button.setEnabled(False)
            self.mic_button.setEnabled(True)
            self.chat_display.append(f'<div style="color: #01579B; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">Mic Only mode ON - Text input disabled</div>')
        else:
            self.message_entry.setEnabled(True)
            self.send_button.setEnabled(True)
            self.mic_button.setEnabled(False)
            self.chat_display.append(f'<div style="color: #01579B; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">Mic Only mode OFF - Text input enabled</div>')
        self.chat_display.ensureCursorVisible()

    def show_chat_interface(self, response):
        self.layout.removeWidget(self.login_widget)
        self.login_widget.deleteLater()
        
        self.chat_widget = QWidget()
        chat_layout = QVBoxLayout(self.chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        header = QLabel("🌿 AI Shop Assistant")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: white; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #26A69A, stop:1 #01579B); padding: 15px;")
        chat_layout.addWidget(header)
        
        logout_button = QPushButton("Logout")
        logout_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        logout_button.setStyleSheet("""
            QPushButton {background-color: #FF7043; color: white; padding: 5px; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #F4511E;}
        """)
        logout_button.clicked.connect(self.logout)
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(logout_button)
        header_layout.addStretch()
        chat_layout.addLayout(header_layout)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 12))
        self.chat_display.setStyleSheet("""
            QTextEdit {background-color: #FFFFFF; border: none; padding: 15px; border-radius: 5px;}
        """)
        chat_layout.addWidget(self.chat_display)
        
        actions_frame = QHBoxLayout()
        for text, cmd in [("🛒 View Cart", "show my cart"), ("💳 Checkout", "checkout"), ("📋 Products", "show me your products")]:
            btn = QPushButton(text)
            btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            btn.setStyleSheet("""
                QPushButton {background-color: #26A69A; color: white; padding: 8px; border: none; border-radius: 5px;}
                QPushButton:hover {background-color: #00897B;}
            """)
            btn.clicked.connect(lambda _, c=cmd: self.submit_message(c))
            actions_frame.addWidget(btn)
        chat_layout.addLayout(actions_frame)
        
        input_frame = QHBoxLayout()
        self.message_entry = QLineEdit()
        self.message_entry.setFont(QFont("Arial", 12))
        self.message_entry.setStyleSheet("padding: 8px; border: 1px solid #26A69A; border-radius: 5px;")
        self.message_entry.returnPressed.connect(self.submit_message)
        input_frame.addWidget(self.message_entry)
        
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {background-color: #FF7043; color: white; padding: 8px; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #F4511E;}
        """)
        self.send_button.clicked.connect(self.submit_message)
        input_frame.addWidget(self.send_button)
        
        self.mic_button = QPushButton("🎤 Mic")
        self.mic_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.mic_button.setStyleSheet("""
            QPushButton {background-color: #01579B; color: white; padding: 8px; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #003F72;}
        """)
        self.mic_button.clicked.connect(lambda: threading.Thread(target=self.record_audio).start())
        self.mic_button.setEnabled(False)
        input_frame.addWidget(self.mic_button)
        
        self.toggle_button = QCheckBox("Mic Only")
        self.toggle_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.toggle_button.setStyleSheet("color: #01579B; padding: 5px;")
        self.toggle_button.stateChanged.connect(self.toggle_input_mode)
        input_frame.addWidget(self.toggle_button)
        
        chat_layout.addLayout(input_frame)
        
        self.layout.addWidget(self.chat_widget)
        self.add_bot_message(response)
        self.add_bot_message({"message": f"How can I assist you today, {self.session.get('name', '')}? 😊", "session": self.session.copy()})
        self.message_entry.setFocus()

    def logout(self):
        try:
            cart = self.session["cart"]
            if not cart and self.customer_id:
                last_order = self.orders_collection.find_one(
                    {"customer_id": self.customer_id},
                    sort=[("created_at", pymongo.DESCENDING)]
                )
                cart = last_order["items"] if last_order and "items" in last_order else []
                total_price = last_order["total_price"] if last_order and "total_price" in last_order else 0
            else:
                total_price = self.session["total_price"]
            if cart:
                cart_summary = "\n".join([f"- {item['product_name']}: {item['quantity']} @ ₹{item['price']}" for item in cart])
                cart_message = f"\nYour cart:\n{cart_summary}\nTotal: ₹{total_price}"
            else:
                cart_message = "\nYour cart is empty."
                total_price = 0
            final_message = f"Logging you out! See you soon! 🌿{cart_message}"
            final_response = {
                "message": final_message,
                "session": {
                    "cart": cart,
                    "name": self.session["name"],
                    "address": self.session["address"],
                    "phone_number": self.session["phone_number"],
                    "paymentConfirmed": self.session["paymentConfirmed"],
                    "total_price": total_price,
                    "input_method": self.session["input_method"]
                }
            }
            self.add_bot_message(final_response)
            print("Logout JSON response:", json.dumps(final_response, indent=2))
            self.cleanup_threads()
            self.layout.removeWidget(self.chat_widget)
            self.chat_widget.deleteLater()
            self.customer_id = None
            self.session = {"cart": [], "name": "", "address": "", "phone_number": "", "paymentConfirmed": False, "total_price": 0, "input_method": "text"}
            self.waiting_for_name = False
            self.waiting_for_address = False
            self.waiting_for_payment_method = False
            self.waiting_for_new_name = False
            self.waiting_for_new_address = False
            self.show_login_screen()
            return final_response
        except Exception as e:
            error_msg = f"Error during logout: {e}"
            self.show_error(error_msg)

    def add_user_message(self, message):
        self.chat_display.append(f'<div style="background-color: #26A69A; color: white; padding: 10px; margin: 5px 20% 5px 0; border-radius: 10px; text-align: right;">{message}</div>')
        self.chat_display.ensureCursorVisible()

    def add_bot_message(self, response):
        self.chat_display.append(f'<div style="background-color: #ECEFF1; color: #01579B; padding: 10px; margin: 5px 0 5px 20%; border-radius: 10px; text-align: left;">{response["message"]}</div>')
        self.chat_display.ensureCursorVisible()
        print("Response JSON:", json.dumps(response, indent=2))

    def submit_message(self, predefined_message=None):
        if self.mic_only_mode:
            self.chat_display.append(f'<div style="color: #ff3333; padding: 10px; margin: 5px 0; border-radius: 10px; text-align: center;">Text input is disabled. Toggle "Mic Only" off to use text.</div>')
            self.chat_display.ensureCursorVisible()
            return
        message = predefined_message or self.message_entry.text().strip()
        if not message:
            return
        self.message_entry.clear()
        self.add_user_message(message)
        self.chat_display.append('<div style="color: #999999; font-style: italic; text-align: center;">Srija is typing...</div>')
        worker = ChatWorker(self, message, "text")
        self.active_threads.append(worker)
        worker.response_signal.connect(self.update_chat_with_response)
        worker.start()

    def update_chat_with_response(self, response):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        self.add_bot_message(response)

    def cleanup_threads(self):
        for thread in self.active_threads[:]:
            thread.stop()
            self.active_threads.remove(thread)
        print("All active threads cleaned up")

    def closeEvent(self, event):
        self.cleanup_threads()
        event.accept()

def main():
    app = QApplication(sys.argv)
    chatbot = VaseegrahVedaChatbot()
    chatbot.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()