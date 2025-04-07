import streamlit as st
from PIL import Image
import base64

def learn_page():
    """Learn section for browsing educational resources"""
    
    # CSS
    st.markdown("""
    <style>
        * {
            font-family: Verdana, sans-serif !important;
        }
        .resource-card {
            background-color: #9fe0df;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .resource-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        .card-title {
            color: #03272e !important;
            font-size: 15px;
            margin-bottom: 10px;
        }
        .card-subtitle {
            color: black;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .resource-tag {
            display: inline-block;
            background-color: #e9ecef;
            color: #495057;
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .video-tag {
            background-color: #ffebee;
            color: #c62828;
        }
        .article-tag {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .beginner-tag {
            background-color: #e3f2fd;
            color: #1565c0;
        }
        .advanced-tag {
            background-color: #fff3e0;
            color: #e65100;
        }
        .resource-button {
            background-color: #408f8c;
            color: white !important;
            text-align: center;
            padding: 8px 15px;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin-top: 10px;
            border: none;
            cursor: pointer;
        }
        .resource-button:hover {
            background-color: #8a3b6a;
        }
        .section-header {
            margin-top: 30px;
            margin-bottom: 20px;
            color: #666;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        /* Custom styling for Streamlit elements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            width: 200px;
            white-space: pre-wrap;
            background-color: #444444;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px; margin-top: -10px">
      <h2 style="margin: 0;">📚 Knowledge Hub</h2>
    </div>
    """, unsafe_allow_html=True)

    # Introduction 
    st.markdown("""
    <div style="background-color: #b16fbd; padding: 15px; border-radius: 30px; margin-bottom: 20px;">
        <h4 style="color: #3b2140;">Welcome to our Learning Center</h4>
        <p style="color: white;">
            Explore our handpicked collection of educational resources to enhance your financial literacy.
            These carefully selected articles and videos will help you get started in your saving journey and let you make informed financial decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different financial topics
    tabs = st.tabs([
        "💰 Basics & Budgeting", 
        "📈 Investing", 
        "🏠 Real Estate", 
        "🔄 Compound Interest",
        "🎓 Tax Planning"
    ])
    
    # Function to create a resource card
    def create_resource_card(title, description, resource_type, level, link, image_url=None):    
        st.markdown(f"""
            <div class="resource-card">
                <h3 class="card-title">{title}</h3>
                <p class="card-subtitle">{description}</p>
                <div>
                    <span class="resource-tag {('video-tag' if resource_type == 'Video' else 'article-tag')}">{resource_type}</span>
                    <span class="resource-tag {('beginner-tag' if level == 'Beginner' else 'advanced-tag')}">{level}</span>
                </div>
                <div style="margin-top: 15px; margin-bottom: 15px;">
                    <img src="{image_url}" alt="{title}" style="display: block; max-width: 200px max-height: 200px; object-fit: cover; border-radius: 5px; margin: auto;">
                </div>
                <div style="margin-top: 15px;">
                   <center><a href="{link}" target="_blank" class="resource-button">Access Resource</a></center>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Basics & Budgeting Resources
    with tabs[0]:
        st.markdown('<h3 class="section-header">Financial Basics & Budgeting</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_resource_card(
                    "Personal Finance 101", 
                    "Learn the fundamentals of managing your personal finances effectively.", 
                    "Article", 
                    "Beginner", 
                    "https://www.investopedia.com/personal-finance-4427760",
                    "https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Creating a Budget That Works", 
                    "A step-by-step guide to creating and sticking to a budget.", 
                    "Video", 
                    "Beginner", 
                    "https://www.youtube.com/watch?v=example1",
                    "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?auto=format&fit=crop&q=80&w=300&h=200"
                )
        
        with col2:
            create_resource_card(
                    "Emergency Fund Guide", 
                    "Why you need an emergency fund and how to build one.", 
                    "Article", 
                    "Beginner", 
                    "https://www.nerdwallet.com/article/banking/emergency-fund-why-it-matters",
                    "https://images.unsplash.com/photo-1579621970590-9d624316904b?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Zero-Based Budgeting Method", 
                    "Advanced budgeting techniques for optimizing your finances.", 
                    "Video", 
                    "Advanced", 
                    "https://www.youtube.com/watch?v=example2",
                    "https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?auto=format&fit=crop&q=80&w=300&h=200"
                ),
    
    # Investing Resources
    with tabs[1]:
        st.markdown('<h3 class="section-header">Investment Education</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_resource_card(
                    "Beginner's Guide to Stock Market", 
                    "Learn the basics of how the stock market works and how to start investing.", 
                    "Article", 
                    "Beginner", 
                    "https://www.investopedia.com/articles/basics/06/invest1000.asp",
                    "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Understanding Mutual Funds", 
                    "A comprehensive guide to mutual funds for beginners.", 
                    "Video", 
                    "Beginner", 
                    "https://www.youtube.com/watch?v=example3",
                    "https://images.unsplash.com/photo-1560520031-3a4dc4e9de0c?auto=format&fit=crop&q=80&w=300&h=200"
                )
        
        with col2:
            create_resource_card(
                    "Portfolio Diversification Strategies", 
                    "Advanced techniques for building a diversified investment portfolio.", 
                    "Article", 
                    "Advanced", 
                    "https://www.morningstar.com/articles/1075188/why-diversification-matters",
                    "https://images.unsplash.com/photo-1559526324-593bc073d938?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Technical Analysis Fundamentals", 
                    "Learn how to analyze stock charts and identify trading opportunities.", 
                    "Video", 
                    "Advanced", 
                    "https://www.youtube.com/watch?v=example4",
                    "https://images.unsplash.com/photo-1543286386-713bdd548da4?auto=format&fit=crop&q=80&w=300&h=200"
                )
    
    # Real Estate Resources
    with tabs[2]:
        st.markdown('<h3 class="section-header">Real Estate Investment</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_resource_card(
                    "First-Time Home Buyer's Guide", 
                    "Everything you need to know before purchasing your first home.", 
                    "Article", 
                    "Beginner", 
                    "https://www.bankrate.com/mortgages/first-time-homebuyer-guide/",
                    "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Rental Property Investment 101", 
                    "How to start investing in rental properties and generate passive income.", 
                    "Video", 
                    "Beginner", 
                    "https://www.youtube.com/watch?v=example5",
                    "https://images.unsplash.com/flagged/photo-1564767609342-620cb19b2357?auto=format&fit=crop&q=80&w=300&h=200"
                )
        
        with col2:
            create_resource_card(
                    "Commercial Real Estate Investment", 
                    "A comprehensive guide to investing in commercial real estate.", 
                    "Article", 
                    "Advanced", 
                    "https://www.forbes.com/sites/forbesrealestatecouncil/2021/04/29/commercial-real-estate-101-how-to-invest-wisely/",
                    "https://images.unsplash.com/photo-1542744095-fcf48d80b0fd?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Real Estate Market Cycles", 
                    "Understanding real estate market cycles and how to time your investments.", 
                    "Video", 
                    "Advanced", 
                    "https://www.youtube.com/watch?v=example6",
                    "https://images.unsplash.com/photo-1460317442991-0ec209397118?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
    # Compound Interest Resources
    with tabs[3]:
        st.markdown('<h3 class="section-header">Compound Interest & Wealth Building</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_resource_card(
                    "The Magic of Compound Interest", 
                    "Understanding how compound interest works and why it's so powerful.", 
                    "Article", 
                    "Beginner", 
                    "https://www.investor.gov/additional-resources/information/youth/teachers-classroom-resources/what-compound-interest",
                    "https://images.unsplash.com/photo-1579621970795-87facc2f976d?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Compound Interest Explained", 
                    "A visual explanation of how compound interest builds wealth over time.", 
                    "Video", 
                    "Beginner", 
                    "https://www.youtube.com/watch?v=example7",
                    "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=300&h=200"
                )
        
        with col2:
            create_resource_card(
                    "Retirement Savings Strategies", 
                    "How to leverage compound interest for retirement planning.", 
                    "Article", 
                    "Advanced", 
                    "https://www.fidelity.com/viewpoints/retirement/how-much-money-do-i-need-to-retire",
                    "https://images.unsplash.com/photo-1634474588707-de99f09285c0?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Optimization Models for Compound Growth", 
                    "Advanced techniques for maximizing the effect of compound interest.", 
                    "Video", 
                    "Advanced", 
                    "https://www.youtube.com/watch?v=example8",
                    "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=300&h=200"
                ),
            
    # Tax Planning Resources
    with tabs[4]:
        st.markdown('<h3 class="section-header">Tax Planning Strategies</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_resource_card(
                    "Tax-Efficient Investing", 
                    "Strategies to minimize taxes on your investment returns.", 
                    "Article", 
                    "Beginner", 
                    "https://www.investopedia.com/articles/stocks/11/intro-tax-efficient-investing.asp",
                    "https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Tax Deductions You Might Be Missing", 
                    "A guide to common tax deductions that many taxpayers overlook.", 
                    "Video", 
                    "Beginner", 
                    "https://www.youtube.com/watch?v=example9",
                    "https://images.unsplash.com/photo-1554224154-22dec7ec8818?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
        with col2:
            create_resource_card(
                    "Advanced Tax Planning Strategies", 
                    "Sophisticated tax planning techniques for high-income individuals.", 
                    "Article", 
                    "Advanced", 
                    "https://www.kiplinger.com/taxes/tax-planning/602464/tax-planning-strategies-for-your-golden-years",
                    "https://plus.unsplash.com/premium_photo-1678567671227-fc52dc1e307f?auto=format&fit=crop&q=80&w=300&h=200"
                )
            
            create_resource_card(
                    "Tax-Loss Harvesting Explained", 
                    "How to use investment losses to offset capital gains tax.", 
                    "Video", 
                    "Advanced", 
                    "https://www.youtube.com/watch?v=example10",
                    "https://images.unsplash.com/photo-1434626881859-194d67b2b86f?auto=format&fit=crop&q=80&w=300&h=200"
                )
    
        
    # Footer
    st.markdown("""
    <div style="background-color: #f0b6c6; padding: 10px; border-radius: 30px; margin-top: 30px; text-align: center;">
        <p style="color: #360714; font-size: 14px;">
            Our team carefully curates these resources to ensure they provide accurate and valuable information.
            However, please note that the content of external websites and videos is beyond our control and may change over time.
            The resources provided are for educational purposes only and do not constitute financial advice.
            Always consult with a qualified financial professional before making important financial decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)